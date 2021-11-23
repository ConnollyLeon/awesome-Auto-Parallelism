# Huawei MindSpore
In this directory, we have examples of **TensorOpt** and **Double Recursive Algorithm**,
which are implemented with Mindspore.

## Prerequisite
1. GCC 7.3.0
2. CUDA 10.1 with cuDNN 7.6.x or CUDA11.1 with cuDNN 8.0.x. 
    
    Installation Instructions: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
    
    Make sure we have CUDA in our environment `PATH` and `LD_LIBRARY_PATH` like `export PATH=/usr/local/cuda-${version}/bin:$PATH`
   and `export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`
   
3. To run experiment on  multiple devices, we need NCCL 2.7.6 with CUDA 10.1 or
    NCCL 2.7.8 with CUDA 11.1
   

Since I have RTX 3090, which only supported by CUDA11.1, I use CUDA11.1 here.

Install Mindspore with conda:
```bash
conda install mindspore-gpu={version} cudatoolkit=11.1 -c mindspore -c conda-forge
```

Check Installation:
```bash
python -c "import mindspore;mindspore.run_check()"
```

Expected Output:
```text
mindspore version: 1.5
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

## Dataset
We use CIFAR-10 to train Resnet-50 model in this experiment.

### Dataset Preparation
> `CIFAR-10` Download Link：<http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>。

On linux machine, we can execute below bash codes to download dataset to directory `cifar-10-batches-bin`。

```bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```

## Run Experiment
### TensorOpt
```bash
bash run_tensoropt.sh {DATA_PATH} 
```

### Double Recursive
```bash
bash run_double_recursive.sh {DATA_PATH}
```