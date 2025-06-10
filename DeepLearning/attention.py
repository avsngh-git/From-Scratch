import torch
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

    