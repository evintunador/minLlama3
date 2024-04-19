# Heavily edited from Meta's original at https://github.com/meta-llama/llama3/blob/main/llama/model.py
import math
from typing import Optional, Tuple

from params import *

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        ).to(args.device)
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        ).to(args.device)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos is not None: # if we're performing inference, use kv caching. it'll be 0 to begin with
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            #print(1, start_pos, seqlen, self.cache_k.shape, xk.shape)
            #print(2, start_pos, seqlen, self.cache_v.shape, xv.shape)
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout_rate = args.dropout_rate

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
        training = False,
    ):
        h = x + F.dropout(self.attention(self.attention_norm(x), freqs_cis, mask, start_pos), p=self.dropout_rate, training=training)
        out = h + F.dropout(self.feed_forward(self.ffn_norm(h)), p=self.dropout_rate, training=training)
        return out


class Llama3(nn.Module):
    def __init__(self, params: ModelArgs, tokenizer):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_seq_len = params.max_seq_len
        self.tokenizer = tokenizer

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, 
            params.vocab_size, 
            bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,)

        mask = torch.full((params.max_seq_len, params.max_seq_len), 
                          float("-inf"), 
                          device=params.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, # specifically for training
                tokens: torch.Tensor, 
                targets: torch.Tensor):
        bsz, seqlen = tokens.shape
        assert tokens.shape == targets.shape
        assert seqlen == self.max_seq_len
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            h = layer(
                h, 
                freqs_cis, 
                self.mask, 
                start_pos = None, 
                training = True
            )
        h = self.norm(h)
        logits = self.output(h).float()

        loss = self.criterion(
            logits.view(bsz * seqlen, self.vocab_size),
            targets.reshape(bsz * seqlen))
        
        return logits, loss

    @torch.inference_mode()
    def forward_inference(self, 
                          tokens: torch.Tensor,
                          start_pos: int,
                          max_context_window: int,
                         ):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = self.mask[:seqlen, :seqlen]
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
        ).type_as(h)

        for layer in self.layers:
            h = layer(
                h, 
                freqs_cis, 
                mask, 
                start_pos = start_pos
            )
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    @torch.inference_mode() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling 
        """
        # Select the last element for each sequence.
        logits = logits[:,-1,:] # (batch_size, input_len, vocab_size) -> (batch_size, vocab_size)
        
        # Apply temperature scaling
        logits.div_(temperature) # (batch_size, vocab_size) / float -> (batch_size, vocab_size)

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along

        # sort the probabilities to for use in top-p & top-k. both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        ### calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

        ### calculating top_k
        # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) 
        # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks
        top_ks_mask = top_ks_mask >= top_k

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        # this trims probs_sort to also fit within our top_k requirement
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization so that total probabilities add up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        
        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        return next_token_id # returns the predicted token

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_gen_len: int = None,
        memory_saver_div: int = 1, # defaults to full max_seq_len**2 memory use. must be power of 2
        temperature: float = 0.6, # default value in meta's code
        top_p: float = 0.9, # default value in meta's code
        top_k: int = tokenizer.vocab_len, # meta's code doesn't bother with topk
    ) -> str: 
        """ Wrapper around sampler() that deals with manipulation of the sequence """
        assert ((memory_saver_div & (memory_saver_div-1)) == 0) & (memory_saver_div > 0), f'memory_saver_div {memory_saver_div} must be power of 2'
        max_context_window = self.max_seq_len // memory_saver_div
        if max_context_window < self.max_seq_len:
            print(f'maximum attention matrix size will be {max_context_window}x{self.max_seq_len} rather than {self.max_seq_len}x{self.max_seq_len}\n')
        
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)
        
        if max_gen_len is None:
            max_gen_len = self.max_seq_len - len(tokens)
        elif max_gen_len + len(tokens) > self.max_seq_len:
            print(f'capping max_gen_len at max_seq_len={self.max_seq_len} including input\n')
            max_gen_len = self.max_seq_len - len(tokens)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=self.params.device)
        tokens = tokens.unsqueeze(0) if len(tokens.shape)==1 else tokens # jic we need to add a batch dimension
        
        start_pos = max(tokens.shape[1] - max_context_window, 0)
        
        for i in range(max_gen_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits = self.forward_inference(
                tokens[:,-max_context_window:],
                start_pos = start_pos,
                max_context_window = max_context_window
            )
            
            next_token = self.Sampler(
                logits = logits,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            
            if tokens.shape[1] >= max_context_window:
                start_pos += 1

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output