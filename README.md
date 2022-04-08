# Concept Explanation

## Data Parallelism (DP)

## Model Parallelism

Model Parallelism has two types: Inter-layer and intra-layer. We note Inter-layer model parallelism as MP, and
intra-layer model parallelism as TP (tensor parallelism).

some researchers may call TP parameter parallelism or intra-layer model parallelism.

Popular intra-model parallelism methods include 2D, 2.5D, 3D model-parallelism as well as Megatron(1D). There are only
few work related to 2D, 2.5D and 3D now (only Colossal-AI).

## Pipeline Parallelism

The partition of PP and MP are similar, but has different executing behaviors. Basically pipeline parallelism has two
families: PipeDream family and GPipe family.

# Published methods of auto-parallelism, including:

I classify parallelism methods according to their partition ways.

## Pipeline Parallelism or Inter-layer Model Parallelism only:

|  Name  | Description | Organization or author | Paper| Framework| Year | Auto Methods
| --- | --- | --- | ---  | --- | --- | --- |
| ColocRL(REINFORCE) | Use reinforce learning to discover model partitions | Google Brain | [mlr.press](http://proceedings.mlr.press/v70/mirhoseini17a/mirhoseini17a.pdf) | Tensorflow | PMLR 70, 2017 | Reinforce
| A hierarchical model for device placement (HDP)| Use Scotch to do graph partitioning | Google |[link](https://openreview.net/pdf?id=Hkc-TeZ0W) | Tensorflow | ICLR 2018 | Reinforce LSTM
| GPipe| No implementation, see torchgpipe | Google | [arxiv](https://arxiv.org/abs/1811.06965) | None| 2018 on arxiv, NIPS2019 | averagely partition or manually
|[torchgpipe](https://github.com/kakaobrain/torchgpipe)| An A GPipe implementation in PyTorch |  UNIST | [arxiv](https://arxiv.org/pdf/2004.09910.pdf) | pytorch | 2020 on arxiv | balance stages by profiling
| GDP | A general deep RL method for automating device placements on arbitrary graphs. Orthogonal to DP,MP,PP | Google| [arxiv](https://export.arxiv.org/pdf/1910.01578.pdf) | Unknown | 2019 on arxiv | Reinforce Transformer
| Pesto | partition model based  on inter-layer model parallelism | Stony Brook University | [acm](https://www3.cs.stonybrook.edu/~anshul/middleware21_pesto.pdf) | Tensorflow | Middleware '21 | integer linear program
| [vPipe](https://github.com/hku-systems/vpipe) | A pipeline only system designed for NAS network. Complementary to hybrid parallelism| HKU | [ieee](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9472938) | PyTorch | TPDS vol.33 no.3 2022 |Swap, Recompute, Partition(SRP) planner. P: Kernighan-Lin algorithm

## Data Parallelism + Pipeline Parallelism (or Inter-layer Model Parallelism):

|  Name  | Description | Organization or author | Paper| Framework| Year | Auto Methods |
| --- | --- | --- | ---  | --- | --- | --- |
| Spotlight| Model device placement as a Markov decision process (MDP). | University of Toronto | [mlr.press](http://proceedings.mlr.press/v80/gao18a/gao18a.pdf) | Unknown|PMLR 80, 2018 | Reinforce LSTM
| Placeto | Looks like Spotlight with MDP, but have different Policy. | MIT |[nips](https://proceedings.neurips.cc/paper/2019/file/71560ce98c8250ce57a6a970c9991a5f-Paper.pdf) | Tensorflow | NIPS 2019 |  Reinforce
|[REGAL](https://github.com/deepmind/deepmind-research/tree/master/regal)|a deep reinforcement learning approach to minimizing the execution cost of neural network computation graphs in an optimizing compiler. |Google|[openreview](https://openreview.net/pdf?id=rkxDoJBYPB) | Unknown |ICLR 2020 |RL with Genetic Algorithm
|[PipeDream](https://github.com/msr-fiddle/pipedream) |This repository contains the source code implementation of PipeDream and PipeDream-2BW | Microsoft Fiddle| [arxiv](https://arxiv.org/pdf/1806.03377.pdf), | PyTorch | 2018 on arxiv, SOSP 2019 | Dynamic Programming with Profile
|PipeDream-2BW | See above one | Microsoft |[arxiv](https://arxiv.org/pdf/2006.09503.pdf), [mlr.press](http://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf)  | PyTorch | PMLR 139, 2021 | Dynamic Programming with Profile
|[DNN-partitioning](https://github.com/msr-fiddle/dnn-partitioning)| published at NeurIPS 2020. | Microsoft Fiddle| [arxiv](https://arxiv.org/pdf/2006.16423.pdf) | proof-of-concept implementation | NIPS 2020 |Dynamic Programming and Integer Programming
|HetPipe| Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism | UNIST | [usenix](https://www.usenix.org/system/files/atc20-park.pdf) | PyTorch (not open sourced) | USENIX 2020 | use CPLEX to solve linear programming problem
|[DAPPLE](https://github.com/AlibabaPAI/DAPPLE) | An Efficient Pipelined Data Parallel Approach for Training Large Model. Succeed from GPipe | Alibaba | [arxiv](https://arxiv.org/pdf/2007.01045.pdf) | DAPPLE | 2020 on arxiv; PPoPP 21 | Dynamic Programming
|[PipeTransformer](https://github.com/Distributed-AI/PipeTransformer) |Automated Elastic Pipelining for Distributed Training of Transformers | University of South  California | [arxiv](https://arxiv.org/pdf/2102.03161.pdf) |PyTorch |  ICML 21 | Dynamic Programming
|[Chimera](https://github.com/Shigangli/Chimera) | Efficiently training large-scale neural networks with bidirectional pipelines | Department of Computer Science, ETH Zurich Switzerland | [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3458817.3476145) | PyTorch | SC 2021 | Performance model with brute force
| TAPP | Use a Seq2Seq based on attention mechanism to predict stage for layers. | Hohai University | [mdpi](https://www.mdpi.com/2076-3417/11/11/4785/pdf) | Unknown |Appl.sci. 2021, 11 | Reinforce Seq2Seq based on attention
|[RaNNC](https://github.com/nict-wisdom/rannc/tree/main) | RaNNC is an automatic parallelization middleware used to train very large-scale neural networks. | DIRECT and University of Tokyo | [arxiv](http://arxiv.org/abs/2103.16063) | PyTorch | IPDPS 2021 | dynamic programming
|[HeterPS](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/framework/fleet/heter_ps)| distributed deep learning with RL based scheduling in heterogeneous environment. | Baidu |  [arxiv](https://arxiv.org/pdf/2111.10635.pdf) | Paddle | 2021 | Reinforce learning based
|[FTPipe](https://github.com/saareliad/FTPipe) | FTPipe can automatically transform sequential implementation into a multi-GPU one. | Technion-Israel Institute of Technology | [usenix](https://usenix.org/system/files/atc21-eliad.pdf) | PyTorch | 2021 | multiprocessor scheduling problem with profiling.

## Data Parallelism + Intra-layer Model Parallelism (or Tensor Parallelism):

|  Name  | Description | Organization or author | Paper| Framework| Year | Auto Methods |
| --- | --- | --- | ---  | --- | --- | --- |
|[OptCNN](https://github.com/flexflow/FlexFlow) | auto parallelism method  for CNN |Zhihao Jia | [mlr.press](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf) | FlexFlow | PMLR 80, 2018 | Dynamic Programming based graph search algorithm
|[FlexFlow](https://github.com/flexflow/FlexFlow) | a deep learning framework that accelerates distributed DNN training by automatically searching for efficient parallelization strategies | Zhihao Jia | [stanford](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf) |FlexFlow, compatible with PyTorch, Keras | SysML 2019 | MCMC
|Tofu| Supporting Very Large Models using Automatic Dataflow Graph Partitioning | New York University | [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3302424.3303953) | Not OpenSourced | Euro-Sys 2019 | same as OptCNN
|[AccPar](https://github.com/linghaosong/AccPar) |Tensor partitioning for heterogeneous deep learning accelerators. | Linghao Song from USC| [usc.edu](http://alchem.usc.edu/portal/static/download/accpar.pdf) | Need Manually Deploy | 2019 on arxiv, HPCA 2020 | Dynamic Programming
|[TensorOpt](https://github.com/mindspore-ai/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel) | Exploring the Tradeoffs in Distributed DNN Training with Auto-Parallelism | CUHK & Huawei |  [arxiv](https://arxiv.org/pdf/2004.10856.pdf) | MindSpore | 2020 on arxiv | Dynamic Programming based graph search algorithm
|[ROC](https://github.com/jiazhihao/ROC) | Another paper from Zhihao, Jia. Designed for GNN | Zhihao Jia | [mlsys](https://proceedings.mlsys.org/paper/2020/file/fe9fc289c3ff0af142b6d3bead98a923-Paper.pdf) | On top of Flexflow  | MLSys 2020 | uses a novel online linear regression model to achieve efficient graph partitioning, and introduces a dynamic programming algorithm to minimize data transfer cost.
|[Double Recursive](https://github.com/mindspore-ai/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel/rec_core) | A Double recursive algorithm to search strategies | Huawei | [link](https://link.springer.com/chapter/10.1007/978-3-030-85665-6_13) | MindSpore | Euro-Par 2021 | Double Recursive
|[PaSE](https://github.com/baidu-research/PaSE) |PaSE uses a dynamic programming based approach to find an efficient strategy within a reasonable time. | Baidu Research | [ieee](https://github.com/baidu-research/PaSE/raw/master/docs/PaSE_ipdps2021.pdf) | prototype | IPDPS 2021 | Dynamic Programming
|P^2| offer a novel syntax-guided program synthesis framework that is able to decompose reductions over one or more parallelism axes to sequences of collectives in a hierarchy- and mapping-aware way |University of Cambridge & DeepMind | [arxiv](https://arxiv.org/pdf/2110.10548.pdf) | Simulation Experiment |2021 on arxiv, MLSys 2022 | Synthesize tool with simulation
|AutoMap| Uses Search and Learn to do find Megatron-like strategies | DeepMind | [arxiv](https://arxiv.org/pdf/2112.02958.pdf) | JAX python API, XLA backend | 2021 on arxiv, NIPS 2021 | Search: Monte Carlo Tree Search; Learn: Interactive Network

## Data Parallelism + Model Parallelism (or Tensor Parallelism) + Pipeline Parallelism:

|  Name  | Description | Organization or author | Paper| Framework| Year | Auto Methods|
| --- | --- | --- | ---  | --- | --- | --- |
|Auto-MAP| It works on HLO IR. Use Linkage Group to prune search space Use DQN RL to search DD, MP, PP stategies. | Alibaba | [arxiv](https://arxiv.org/pdf/2007.04069.pdf) | RAINBOW DQN | 2020 | Reinforce Learning
|[Piper](https://github.com/msr-fiddle/piper) | This code package contains algorithms (proof-of-concept implementation) and input files (profiled DNN models / workloads) from the paper "Piper: Multidimensional Planner for DNN Parallelization" published at NeurIPS 2021. An extension of DNN partitioning| Microsoft Fiddle| [link](https://www.microsoft.com/en-us/research/publication/piper-multidimensional-planner-for-dnn-parallelization/) | proof-of-concept implementation | NIPS 2021 | two-level dynamic programming
|[GSPMD](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla) |a system that uses simple tensor sharding annotations to achieve different parallelism paradigms in a unified way | Google | [arxiv](https://arxiv.org/pdf/2105.04663.pdf) | Tensorflow XLA | 2021 | sharding propagation
|[DistIR](https://github.com/microsoft/dist-ir) | Horizontal TP. An intermediate representation and simulator for efficient neural network distribution | Stanford University & Microsoft Fiddle| [arxiv](https://arxiv.org/abs/2111.05426) | PyTorch | MLSys 2021 | Grid-Search Simulator
|Neo | A software-hardware co-designed system for high-performance distributed training of large-scale DLRM.  | Facebook | [arxiv](https://export.arxiv.org/pdf/2104.05158.pdf) | PyTorch | 2021 | 1. Greedy 2. Karmarker-Karp Algorithm
|Adaptive Paddle| Elastic training, fault tolerant, Cost-model based Sharding propagation |Baidu | [arxiv](https://arxiv.org/pdf/2112.02752.pdf) | Paddle | 2021 | Cost model based. Details un-given.
|[Alpa](https://github.com/alpa-projects/alpa) | Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | UC Berkley, Google, etc. | [arxiv](https://arxiv.org/pdf/2201.12023.pdf) | Jax, XLA | 2022 | Integer Linear for Intra, Dynamic programming for inter

## Other Interesting automatic work

|  Name  | Description | Organization or author | Paper| Framework| Year | Auto Methods|
| --- | --- | --- | ---  | --- | --- | --- |
|[TASO](https://github.com/jiazhihao/TASO) | automatically optimize DNN computation with graph substitution |  Zhihao Jia |

---

# Classify with Machine-Learning Based Methods and Classic Algorithm Based Methods

## Machine-Learning Based Methods

| Name | Method Type | Parallelism | Year |
| --- | --- | --- | ---  |
| ColocRL | Reinforcement | MP | 2017 | 
| HDP | Reinforcement | MP | 2018 |  
| GDP | Reinforcement | MP | 2019 |  
| REGAL | Reinforcement | MP | 2020 |  
| TAPP | Reinforcement | DP+PP | 2021 |
| Spotlight | Reinforcement | DP+MP | 2018 |
| Placeto | Reinforcement | DP+MP | 2019 |  
| HeterPS | Reinforcement | DP+PP | 2021 | 
| AutoMap | Deep Learning to predict rank | DP+TP | 2021 | 
| Auto-MAP | Reinforcement | DP or TP or PP | 2020 | 
| FlexFlow | MCMC | DP+TP  | 2019 | 
| ROC | uses a novel online linear regression model to achieve efficient graph partitioning, and introduces a dynamic programming algorithm to minimize data transfer cost. | DP+TP  | 2020 | 

## Classic Algorithm Based Methods

| Name | Method Type | Parallelism | Year |
| --- | --- | --- | ---  |
| Pesto | integer linear | MP | 2021 | 
| vpipe | SRP algorithm + KL (DP) | PP | 2022 | 
| PipeDream | dynamic programming | DP+PP | 2019 | 
| DNN-partitioning | dynamic programming + integer programming | DP+PP | 2020 |
| PipeDream-2BW | dynamic programming | DP+PP | 2021 |
| HetPipe | dynamic programming | DP+PP | 2020 |   
| DAPPLE | dynamic programming | DP+PP | 2021 | 
| PipeTransformer | dynamic programming | DP+PP | 2021 | 
| Chimera | Grid-Search| DP+PP | 2021 | 
| RaNNC | dynamic programming | DP+PP | 2021 |  
| FTPipe | Multiprocessor scheduling problem with profiling | DP+PP | 2021 |
| OptCNN | dynamic programming | DP+TP | 2018 | 
| Tofu | dynamic programming | DP+TP | 2019 | 
| AccPar | dynamic programming | DP+TP | 2020 | 
| TensorOpt | dynamic programming | DP+TP | 2020 | 
| Double Recursive | Double recursive  | DP+TP | 2021 |
| PaSE | dynamic programming | DP+TP | 2021 |
| P^2 | Synthesize tool with simulation | DP+TP | 2021 |  
| Piper | two-level dynamic programming | DP+TP+PP | 2021 |
| GSPMD | heuristic-propagation | DP+TP+PP | 2021 | 
| DistIR | grid search | DP+TP+PP | 2021 | 
| Neo | Greedy + Karmarker-karp algorithm | DP+TP+PP | 2021 |
| Alpa | Integer programming + Dynamic Programming | DP+TP+PP | 2022 |

---

## Pictures

### REINFORCE

![img.png](Image/overall/reinforce.png)

### Spotlight

![img.png](Image/overall/spotlight.png)

### GPipe

![img.png](Image/overall/gpipe.png)

### GDP

![img.png](Image/overall/gdp.png)

### Placeto

![img.png](Image/overall/placeto.png)

### REGAL

![img.png](Image/overall/REGAL.png)

# News

2021.12.9 DeepMind proposes Gopher, a 280 billion parameter transformer language model. Trained by 4096 16GB
TPUv3. [link](https://deepmind.com/blog/article/language-modelling-at-scale)

2021.12.8 Baidu and Peng Cheng proposes Wenxin (文心), a 260 billion parameter knowledge-aware pretrained model (a.k.a.
ERNIE 3.0 Titan). Trained with Adaptive Paddle in the Table above.

2021.10.26 Inspur formally proposes 245.7 billion parameter on AICC 2021.s