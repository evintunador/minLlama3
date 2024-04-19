import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass
from typing import Optional

# imports for the tokenizer
from tiny_shakespeare_tokenizer import *

tokenizer = get_tokenizer(size = 512) # size options are 128, 256, 512 and 1024

@dataclass
class ModelArgs:
    dim: int = 128 # 4096
    n_layers: int = 8 # 32
    n_heads: int = 4 # 32
    n_kv_heads: Optional[int] = 1 # None
    vocab_size: int = tokenizer.vocab_len # -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000 # 500000
    max_batch_size: int = 32
    max_seq_len: int = 512 # 2048
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout_rate: float = 0.1