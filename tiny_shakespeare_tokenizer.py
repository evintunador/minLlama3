# Importing pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

# used for the tokenizer
import pickle
import os

# Load the tokenizer data using pickle
with open('./tokenizers/tokenizer.model', 'rb') as f:
    loaded_tokenizer_data = pickle.load(f)

# Extract the stoi mapping and merges from the loaded data
loaded_stoi = loaded_tokenizer_data['stoi']
loaded_merges = loaded_tokenizer_data['merges']

class SimpleTokenizer:
    def __init__(self, stoi, merges):
        self.stoi = stoi
        self.merges = merges
        self.itos = {i: s for s, i in stoi.items()}  # Inverse mapping for decoding

        self.vocab_len = len(stoi) + len(merges)

    def encode(self, text):
        # Convert the text to a list of token IDs, using space for unknown characters
        tokens = [self.stoi.get(c, self.stoi[' ']) for c in text]

        # Perform merging with the possibility of nested merges
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                # Replace the current pair with its merged token
                merged_token = self.merges[pair]
                tokens[i] = merged_token
                del tokens[i + 1]

                # Move back to handle possible nested merges
                if i > 0:
                    i -= 1
            else:
                i += 1

        return tokens

    def decode(self, tokens):
        def expand_token(token):
            # Base case: if the token is a direct mapping, return its character
            if token in self.itos:
                return self.itos[token]
            # Recursive case: if the token is a merged token, expand its constituents
            elif token in self.merges.values():
                pair = next(key for key, value in self.merges.items() if value == token)
                return ''.join(expand_token(t) for t in pair)
            # Fallback for unknown tokens
            else:
                return ''

        # Decode each token in the list, handling nested merges recursively
        return ''.join(expand_token(token) for token in tokens)

tokenizer = SimpleTokenizer(loaded_stoi, loaded_merges)