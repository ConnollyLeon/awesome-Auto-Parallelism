#DistIR

# Summary
Users don't need to write dist-ir code. Users can write forward-only code
like PyTorch, and then export it to ONNX or XLA. Then import the ONNX or XLA 
model to DistIR. DistIR can then use grid-search to find the best parallelism 
strategies.

There are two ways to measure a strategy in DistIR now, which are Simulator and 
PyTorch running time. Simulator use a CostModel to simulate the computation and 
communication time. PyTorch running time uses `multiprocessing` library to profile
its executing time.

Currently, DistIR only support grid search on data-parallelism, tensor-parallleism 
and 1F1B PipeDream pipeline-parallelism, as well as the num of micro-batches.

In the future, DistIR may support ZeRO and overlapping communication and computation. 


# Experiment
I  ran this experiment on a Ubuntu single node with 8 V100.
## Prerequisite
To run this example, Make sure you have Anaconda installed, and 
`git lfs` is runnable.

To install git lfs on Ubuntu, you can run:
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

### Step1: Environment Settings
```bash
conda create -n distir python=3.8
conda activate distir
```

### Step 2: Clone repository
```bash
git clone https://github.com/microsoft/dist-ir.git
cd dist-ir
```

### Step 3: Install dependencies
```bash
pip install pylint, pytest, black

# DistIR Dependencies
pip install -r requirements.txt 
```

### Step 4: Prepare GPT2-10 ONNX model
```bash
pushd /tmp 
git clone https://github.com/onnx/models.git
pushd models
git checkout bb0d4cf3d4e2a5f7376c13a08d337e86296edbe8h
git lfsm pull --include="text/machine_comprehension/gpt-2/model/gpt2-10.onnx" --exclude ""
popd
popd
mv /tmp/models/text/machine_comprehension/gpt-2/model/gpt2-10.onnx ./
```

### Step 5: Check formatting (black)
```bash
balck --diff --check .
```

### Step 6: Install dist-ir
```bash
python setep.py install
```

### Step 7: Run pytest
```bash
python -m pytest
```
