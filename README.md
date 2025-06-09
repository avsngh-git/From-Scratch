# From-Scratch
Implementation of various concepts from Machine Learning in code from scratch. I will be using pytorch for some stuff but the math will be from scratch.

## Work in Progress
Batch Normalization

## Finished
1. Batch Normalization - implemented batch normalization using batch_norm fucntion which includes all the math of the operation and BatchNorm class which uses the function and does other bookeeping. It is designed to work on tabular and image data. No automatic shape inference. It also saves Mean and Var over the whole dataset for use during inference.