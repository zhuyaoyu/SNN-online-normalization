import datetime
import os
import time
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp
from models import spiking_resnet_SEW, spiking_resnet_NF, spiking_vgg
from modules import neurons, surrogate, neuron_spikingjelly
import config
from datasets.data import get_dataset
import argparse
import math
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.utils.data as data
import seaborn as sn
import matplotlib.pyplot as plt

_seed_ = 2023
import random
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

def is_dynamic(dataset):
    return dataset.lower() in ['cifar10dvs', 'dvsgesture']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='config .yaml file')
    parser.add_argument('-ckpt', type=str, help='chekpoint path to resume')
    
    cfg = parser.parse_args()
    config.parse(cfg.config)
    config.args.ckpt = cfg.ckpt
    config.args.record_v_stat = True
    args = config.args

    # print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    num_classes, trainset, testset = get_dataset(args)
    test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, pin_memory=True)

    c_in = 2 if is_dynamic(args.dataset) else 3
    if args.dataset != 'imagenet':
        # neuron0 = neurons.OnlinePLIFNode if not args.BPTT else neuron_spikingjelly.ParametricLIFNode
        neuron0 = neurons.OnlineLIFNode if not args.BPTT else neurons.MyLIFNode
        net = spiking_vgg.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, neuron_dropout=args.drop_rate, fc_hw=1, BN=args.BN, weight_standardization=args.WS)
    else:
        neuron0 = neurons.OnlineLIFNode if not args.BPTT else neurons.MyLIFNode
        assert(args.model_type is not None and args.model_type.upper() in ['SEW', 'NF'])
        model_set = spiking_resnet_SEW if args.model_type.upper() == 'SEW' else spiking_resnet_NF
        net = model_set.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, drop_rate=args.drop_rate, stochdepth_rate=args.stochdepth_rate, neuron_dropout=0.0, zero_init_residual=False)
    #print(net)
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()


    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #start_epoch = checkpoint['epoch'] + 1
        #max_test_acc = checkpoint['max_test_acc']

    criterion_mse = nn.MSELoss(reduce=True)

    for epoch in range(1):
        start_time = time.time()

        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(test_data_loader))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
    
        mems_all = []
        with torch.no_grad():
            for frame, label in test_data_loader:
                batch_idx += 1
                frame = frame.float().cuda()
                label = label.cuda()
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    input_frame = frame[:, t] if is_dynamic(args.dataset) else frame
                    # print(input_frame.shape, is_dynamic(args.dataset))
                    if t == 0:
                        out_fr = net(input_frame, init=True, save_spike=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame, save_spike=True)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss
                    mems_t = net.get_mem()
                    if len(mems_all) == 0:
                        for mem in mems_t:
                            mems_all.append(mem.reshape(1,-1).detach().cpu().numpy())
                    else:
                        for i, mem in enumerate(mems_t):
                            mems_all[i] = np.concatenate([mems_all[i], mem.reshape(1,-1).detach().cpu().numpy()], axis=0)

                if args.BPTT:
                    net.reset_v()

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(total_loss, input_frame.size(0))
                top1.update(prec1.item(), input_frame.size(0))
                top5.update(prec5.item(), input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=len(test_data_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
                break
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples

        pal = sn.blend_palette([sn.desaturate("royalblue", 0), "royalblue"], config.args.T)
        for i, mem in enumerate(mems_all):
            for t in range(len(pal)):
                mem_t, color = mem[t], pal[t]
                sn.kdeplot(mem_t, bw_adjust=.5, cut=0, color=color, label=f"t={t}", fill=True)
            plt.savefig(f"figs/layer_{i}.png")
        
        cfgname = cfg.config.split('/')[-1]
        os.makedirs("stats", exist_ok=True)
        # outfile = os.path.join("stats", cfgname[:cfgname.find('.')] + '_membrane_potential.npz')
        # np.savez(outfile, membrane_potential=np.array(mems_all))

if __name__ == '__main__':
    main()
