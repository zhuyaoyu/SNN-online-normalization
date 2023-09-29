# Online Stabilization of Spiking Neural Networks

Dependencies:

- python 3
- pytorch and torchvision (pytorch 2.0 preferred)
- NVIDIA GPU + CUDA
- Other dependencies are provided in requirements.txt, which can be installed by pip install -r requirements.txt

### Training

First you may change the paths provided in yamls/your_config.yaml, then you can run the code either with single GPU or Multi GPU as follows:

#### Single GPU

```
CUDA_VISIBLE_DEVICES=0 python train.py -config yamls/CIFAR100.yaml
```

#### Multi GPU

```text
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29000 train.py -config yamls/ImageNet.yaml
```