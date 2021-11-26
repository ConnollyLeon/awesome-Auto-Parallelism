# FlexFlow

## Summary
FlexFlow uses MCMC algorithm to automatically search strategies for each
operator.

Currently FlexFlow has a lack of documents to instruct users how to use it well.

## Installation

Follow instructions on [FlexFlow/INSTALL.md](https://github.com/flexflow/FlexFlow/blob/master/INSTALL.md)

1. Clone code from github
   ```bash
    git clone --recursive https://github.com/flexflow/FlexFlow.git
   ```
2. edit `config/config.linux` to fit your demands.

3. build FlexFlow
   ```bash
    mkdir build
    cd build
    ../config/config.linux
    make
   ```
   
4. python library dependencies
```bash
pip install cffi
pip install keras-preprocessing
pip install pillow
```

### Some problems I met while installing

1. could not find cmake

   Install cmake using this line of code on ubuntu
    ```bash
   sudo apt-get install cmake
    ```

2. could not find hdf5

   Install using this line of code on ubuntu
    ```bash
   sudo apt-get install libhdf5-serial-dev
    ```

## Experiments
follow the [autotune tutorial](https://flexflow.ai/search/)

execute cmd like 
```bash
./dlrm -ll:gpu 4 -ll:fsize 12000 -ll:zsize 20000 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --batch-size 1024 --budget 1000
```
to generate the stategies using MCMC.

### DLRM example
using 4 GPU, one strategy of an Embedding layer looks like
```text
[Dense_100] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_101] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_102] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Embedding_103] num_dims(2) dims[1,1] device_ids[3]
[Embedding_104] num_dims(2) dims[1,1] device_ids[2]
[Embedding_105] num_dims(2) dims[1,1] device_ids[2]
[Embedding_106] num_dims(2) dims[1,1] device_ids[0]
[Embedding_107] num_dims(2) dims[1,1] device_ids[1]
[Embedding_108] num_dims(2) dims[1,1] device_ids[1]
[Embedding_109] num_dims(2) dims[1,1] device_ids[3]
[Embedding_110] num_dims(2) dims[1,1] device_ids[0]
[Concat-111] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_112] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_113] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_114] num_dims(2) dims[1,4] device_ids[0,1,2,3]
[Dense_115] num_dims(2) dims[1,4] device_ids[0,1,2,3]
```
Each line describes the parallelization configuration for one operator: 
`dims` indicates the degree of parallelism for each dimension, and `device_ids` shows the device assignment for each task within an operator. 

It seems quite rational to distribute different embedding tables to every device and
partition a Dense network's weights matrix to every device.
