import torch
import math
from torch import nn


def masked_softmax(X, valid_lens):
    """applies masked softmax function to deal with sequences of variable length in a batch"""

    def sequence_mask(X, valid_lens, value=0):
        """creates a mask for dummy tokens and replaces their value with a high negative 
        number so that softmax takes it to zero"""
        max_len = X.size(1)
        mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None,:]<valid_lens[:,None]
        X[~mask] = value
        return X
    
    
    if valid_lens == None: 
        return nn.functional.softmax(X, dim=1)
    else:
        shape = X.shape
        if valid_lens.dim() ==1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens=torch.reshape(-1)

    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention with batch matrix multiplication and dropout for regularization"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Query shape = (batch_size, no. of queries, d)
    # Key = (batch_size, #keys, d)
    # Values = (batch_size, #k-v pairs, value dimension)
    # valid_lens = (batch_size,) or (batch_size, #queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.dim()
        scores = torch.bmm(queries, keys.transpose(1,2))/math.sqrt(d) #swapping only last 2 dimensions in transpose
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


