# Torchgpipe

torchgpipe is a library that support only Pipeline parallelism.
It can partition a model according to "memory of each layer" or 
"elapsed time of each layer". 

However, it is still a coarse-grained method that has many bubbles in 
the pipeline.

Note:
1. It only supports nn.Sequential model, which means we may need to rewrite
 our model if we want to use it.
   
2. It is coarse-grained. It partitions stages into parts of the Sequential model.

3. Easy to use. Can run with only one process.