# Distributed Face_Recognition Training in Pytorch

## Requirements

To avail the latest features of PyTorch, we have upgraded to version 1.12.0.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (torch>=1.12.0).
- `pip install -r requirement.txt`.
  
## How to Training

To train a model, execute the `train.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.

### 1. To run on one GPU:

```shell
python train.py
```

Note:   
It is not recommended to use a single GPU for training, as this may result in longer training times and suboptimal performance. For best results, we suggest using multiple GPUs or a GPU cluster.  


### 2. To run on a machine with 8 GPUs:

```shell
torchrun --nproc_per_node=8 train.py
```

### 3. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train.py
```

Node 1:
  
```shell
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train.py
```
