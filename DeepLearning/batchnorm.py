import torch

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


