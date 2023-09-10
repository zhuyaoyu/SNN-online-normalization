CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29000 train.py -config yamls/ImageNet_static_th.yaml;
shutdown -h now;