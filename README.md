# Online-training

### Single GPU

```
CUDA_VISIBLE_DEVICES=0 python train.py -config yamls/CIFAR100.yaml
```

### Multi GPU

```text
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29000 train.py -config yamls/ImageNet.yaml
```