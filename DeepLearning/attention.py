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



class AdditiveAttention(nn.Module):
    """ Additive attention"""
    def __init__(self, num_hidden, dropout):
        self.W_q = nn.LazyLinear(num_hidden, bias=False)
        self.W_k = nn.LazyLinear(num_hidden, bias=False)
        self.W_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, valid_lens):
        #mat mul with weight matrices to get the query and key vectors in same dimensions for addition
        queries, keys = self.W_q(queries), self.W_k(keys)

        #queries.shape = (batch, no. of queries, dimensions of the query)
        # keys.shape = (batch, no. of keys, dimensions of the keys)
        # dimensions of keys and queries are the same after multiplications with W but
        # no. of keys and no. of queries fall in the same dim, we dont want to add these but concatenate these
        # thus we add 1 dimension in keys and queries each using unsqueeze and then add

        summed = queries.unsqueeze(2) + keys.unsqueeze(1)
        summed = torch.tanh(summed)
        scores = self.W_v(summed).squeeze(-1) #squeeze(-1) removes the last dimension if it is of size 1
        # after mat mul with W_v and squeeze of last dim teh shape is (batch, no. of queries, no. of key value pairs)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


        


