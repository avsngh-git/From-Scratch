# From-Scratch
Implementation of various concepts from Machine Learning in code from scratch. I will be using pytorch for some stuff but the math will be from scratch.

## Work in Progress
* Attention 

## Finished
 1. #### Batch Normalization 
    implemented batch normalization using batch_norm function which includes all the math of the operation and BatchNorm class which uses the function and does other bookeeping. It is designed to work on tabular and image data. No automatic shape inference. It also saves Mean and Var over the whole dataset for use during inference.

 2. #### Attention
    2.1 masked_softmax - For dealing with sequences of variable size and esuring that attension does not pay any heed to dummy tokens used to makes all the sizes equal. 
    2.2 Scaled Dot Product Attention class implemented which uses masked_softmax to get attention scores, it also makes use of batch matrix multiplication and dropout for regularization.
    2.3 Additive Attention class implemented which is used when queries and keys do not have the same dimensions.
    2.4 Multi Head Attention using Scaled Dot Product Attention class implemented
    2.5 