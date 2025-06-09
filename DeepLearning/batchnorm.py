import torch
from torch import nn

def batch_norm(gamma, beta, x, epsilon, momentum, moving_mean, moving_var):
    """
    math of batch normalization to the input tensor x."""
    if not torch.is_grad_enabled(): # checking if in traning mode
        x_hat = (x-moving_mean)/torch.sqrt(moving_var+epsilon) #if not

    else:
        assert x.dim()==2 or x.dim()==4 # input is either 2D data or 4D images with batch of images as 4th dimension
        if x.dim() == 2:
            mean = x.mean(dim=0)
            var = ((x-mean)**2).mean(dim=0)
        else:
            mean = x.mean(dim=(0,2,3),keepdim=True) # when dim is 4, we compute mean and variance over batch and spatial dimensions
            var = ((x-mean)**2).mean(dim=(0,2,3), keepdim=True)
        
        x_hat = (x-mean)/torch.sqrt(var+epsilon)
        moving_mean = (1.0 -momentum)*mean + momentum*moving_mean # momentum ensures that moving mean and var are updated gradually
        moving_var = (1-momentum)*var + momentum*moving_var
    Y = gamma*x_hat+beta
    return Y, moving_mean.detach(), moving_var.detach() # detach to avoid tracking gradients for moving mean and var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims ==2:
            shape = (1, num_features)  # setup shape for proper broadcasting depending on if its a picture or a normal data
        else:
            shape = (1, num_features, 1, 1)

        # initialize the scale and shift parameters to 1 and 0
        # initialize them as Parameter class to tell pytorch that these are learnable parameters
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # now initialize buffers for mean and var for use during inference
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        #If X is not on the main memory
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # Get the normalized output and save moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(self.gamma, self.beta, X, 1e-5, 0.9)
        return Y