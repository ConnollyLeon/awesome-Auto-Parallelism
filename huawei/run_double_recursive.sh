#!/bin/bash
# applicable to GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH"
echo "For example: bash run_gpu.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf double_recursive
mkdir double_recursive
cp ./double_recursive ./resnet.py ./double_recursive
cd ./tensor_opt
echo "start training"
mpirun -n 2 pytest -s -v ./double_recursive > train.log 2>&1 &
