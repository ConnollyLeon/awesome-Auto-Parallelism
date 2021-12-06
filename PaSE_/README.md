# PaSE

PaSE is proposed by Baidu-Research.

Currently, it is implemented by python. So the efficiency may not be high enough.

## Advantages:

1. PaSE support brute-force search and dynamic-programming based search.
2. Its complexity is O（|V|^2 K^(M+1)）, where V is the num of vertices, K is the configurable strategies of a single
   layer, M is the maximum num of dependent set.

## Limitations:

1. Do not support inter-layer pipeline parallelism. Prevent us from overlapping computation and communications between
   different layers.

2. Not beneficial when the graph is dense, like `DenseNet`, making the `M` size cannot be reduced to an ideal value, and
   leading to high runtime overhead.

3. Though this method is applicable to `heterogeneous` architecture, it does not explicitly include heterogeneity into
   the cost model.
   
4. Ignores several low level details such as `cache effects` in cost model.

## How  to Use

### git clone

```bash
git clone https://github.com/baidu-research/PaSE.git
```

### create virtual environment and install required libraries.

```bash
> python3 -m venv ~/env/pase
> source ~/env/pase/bin/activate
> pip install -r requirements.txt
```

### Execute

```bash
> python3 ./scheduler.py --help
usage: scheduler.py [-h] [-p PROCS] [-b BATCH] [-m MODEL] [-g {alexnet,resnet101,inception3,rnnlm,transformer}] [-a {0,1}] [--profile] [--measure] [-d] [--algo {0,1}]
```

```bash
optional arguments:
  -h, --help            show this help message and exit
  -p PROCS, --procs PROCS
                        No. of processors. (Default: 32)
  -b BATCH, --batch BATCH
                        Batch size. (Default: 128)
  -m MODEL, --model MODEL
                        Model size. (Default: 128)
  -g {alexnet,resnet101,inception3,rnnlm,transformer}, --graph {alexnet,resnet101,inception3,rnnlm,transformer}
                        Neural net graph. (Default: 'alexnet')
  --flops FLOPS         Peak FLOPS of each device in TFLOPS. (default: 10.0)
  --bw BW               Peak inter-connection bandwidth in GBytes/sec (default: 16.0)
  --profile             Turn on/off profiling.
  --measure             Turn on/off measurement.
  -d, --dump-graph      Dump the graph in dot format to the file graph.dot in the working directory.
  --algo {0,1}          Algorithm to be used to compute strategy (Default: 0).
```