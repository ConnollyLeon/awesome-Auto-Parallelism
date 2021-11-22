Published methods of auto-parallelism, including:

## Data Parallelism + Pipeline Parallelism:

|  Name  | Description | Organization or author | Paper| Framework| 
| --- | --- | --- | ---  | --- |
|[DAPPLE](https://github.com/AlibabaPAI/DAPPLE) | An Efficient Pipelined Data Parallel Approach for Training Large Model | Alibaba | [arxiv](https://arxiv.org/pdf/2007.01045.pdf) | DAPPLE
|GPipe| No implementation, see torchgpipe | Google | [arxiv](https://arxiv.org/abs/1811.06965)
|[torchgpipe](https://github.com/kakaobrain/torchgpipe)| An A GPipe implementation in PyTorch |  UNIST | [arxiv](https://arxiv.org/pdf/2004.09910.pdf) | pytorch
|[PipeDream](https://github.com/msr-fiddle/pipedream) |This repository contains the source code implementation of PipeDream and PipeDream-2BW | Microsoft | [arxiv](https://arxiv.org/pdf/1806.03377.pdf), [2BW](https://arxiv.org/pdf/2006.09503.pdf)| Fiddle
|[Chimera](https://github.com/Shigangli/Chimera) | efficiently training large-scale neural networks with bidirectional pipelines | Department of Computer Science, ETH Zurich Switzerland | [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3458817.3476145) | PyTorch
|[RaNNC](https://github.com/nict-wisdom/rannc/tree/main) | RaNNC is an automatic parallelization middleware used to train very large-scale neural networks. | DIRECT and University of Tokyo | [arxiv](http://arxiv.org/abs/2103.16063) | PyTorch

## Data Parallelism + Model Parallelism (or Tensor Parallelism):

|  Name  | Description | Organization or author | Paper| Framework| 
| --- | --- | --- | ---  | --- |
|[OptCNN](https://github.com/flexflow/FlexFlow) | auto parallelism method  for CNN |Zhihao Jia | [mlr.press](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf) | FlexFlow
|[FlexFlow](https://github.com/flexflow/FlexFlow) | a deep learning framework that accelerates distributed DNN training by automatically searching for efficient parallelization strategies | Zhihao Jia | [stanford](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf) |FlexFlow, compatible with PyTorch, Keras
|[AccPar](https://github.com/linghaosong/AccPar) |Tensor partitioning for heterogeneous deep learning accelerators. | Linghao Song from USC| [usc.edu](http://alchem.usc.edu/portal/static/download/accpar.pdf) | Need Manually Deploy
|[TensorOpt](https://github.com/mindspore-ai/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel) | Exploring the Tradeoffs in Distributed DNN Training with Auto-Parallelism | CUHK & Huawei |  [arxiv](https://arxiv.org/pdf/2004.10856.pdf) | MindSpore
|Tofu| Supporting Very Large Models using Automatic Dataflow Graph Partitioning | New York University | [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3302424.3303953) | Not OpenSourced
|[Double Recursive](https://github.com/mindspore-ai/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel/rec_core) | A Double recursive algorithm to search strategies | Huawei | [link](https://link.springer.com/chapter/10.1007/978-3-030-85665-6_13) | MindSpore 

