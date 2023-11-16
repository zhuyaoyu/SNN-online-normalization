import datetime
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
from models import spiking_resnet_SEW, spiking_resnet_NF, spiking_vgg
from modules import layers, neurons, surrogate, neuron_spikingjelly
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets.data import get_dataset
import config
import argparse
import math
import torch.utils.data as data
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore')


def init_seeds(_seed_):
    random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)


def is_dynamic(dataset):
    return dataset.lower() in ['cifar10dvs', 'dvsgesture']


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='config .yaml file')
    parser.add_argument('-ckpt', type=str, help='chekpoint path to resume')

    
    cfg = parser.parse_args()
    if cfg.local_rank >= 0:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend=cfg.dist)
        multigpu = True
    else:
        multigpu = False
    init_seeds(1)
    config.parse(cfg.config)
    args = config.args

    # print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    num_classes, trainset, testset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset) if multigpu else None
    if train_sampler is None:
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.b, num_workers=8, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.b, num_workers=8, pin_memory=True)
    # train_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j, pin_memory=True)
    test_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, pin_memory=True)

    # TODO: LIF or PLIF? should we add this choice?
    c_in = 2 if is_dynamic(args.dataset) else 3
    print(f'args.tau = {args.tau}')
    if args.dataset != 'imagenet':
        # neuron0 = neurons.OnlinePLIFNode if not args.BPTT else neuron_spikingjelly.ParametricLIFNode
        neuron0 = neurons.OnlineLIFNode if not args.BPTT else neurons.MyLIFNode
        net = spiking_vgg.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, neuron_dropout=args.drop_rate, fc_hw=1, BN=args.BN, weight_standardization=args.WS, light_classifier=args.light_classifier)
    else:
        neuron0 = neurons.OnlineLIFNode if not args.BPTT else neurons.MyLIFNode
        assert(args.model_type is not None and args.model_type.upper() in ['SEW', 'NF'])
        model_set = spiking_resnet_SEW if args.model_type.upper() == 'SEW' else spiking_resnet_NF
        net = model_set.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, drop_rate=args.drop_rate, stochdepth_rate=args.stochdepth_rate, neuron_dropout=0.0, zero_init_residual=False)
    #print(net)
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()
    assert(args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    criterion_mse = nn.MSELoss(reduce=True)

    net.eval()
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(train_loader))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0
        for frame, label in train_loader:
            batch_idx += 1
            frame = frame.float().cuda()
            t_step = args.T_train if args.T_train is not None else args.T

            if is_dynamic(args.dataset):
                frame = frame.transpose(0,1)
                if args.T_train and args.T_train != args.T:
                    sec_list = np.random.choice(frame.shape[0], args.T_train, replace=False)
                    sec_list.sort()
                    frame = frame[sec_list]
                    t_step = args.T_train

            label = label.cuda()

            batch_loss = 0
            if args.BPTT:
                bptt_loss = 0
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            for t in range(t_step):
                input_frame = frame[t] if is_dynamic(args.dataset) else frame
                amp_context = amp.autocast if args.amp else nullcontext
                with amp_context():
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                
                ddp_context = net.no_sync if cfg.local_rank != -1 and t == t_step != 0 else nullcontext
                with ddp_context():
                    if args.amp:
                        scaler.scale(loss).backward()
                    else:
                        if not args.BPTT:
                            loss.backward()
                        else:
                            bptt_loss += loss

                batch_loss += loss.item()
                train_loss += loss.item() * label.numel()
            if args.BPTT:
                bptt_loss.backward()
                optimizer.step()
                net.reset_v()
            else:
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(batch_loss, input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))


            train_samples += label.numel()
            train_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=len(train_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()

        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar.finish()
        bar = Bar('Processing', max=len(test_loader))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        with torch.no_grad():
            for frame, label in test_loader:
                batch_idx += 1
                frame = frame.float().cuda()
                label = label.cuda()
                t_step = args.T
                total_loss = 0
                if is_dynamic(args.dataset):
                    frame = frame.transpose(0,1)

                for t in range(t_step):
                    input_frame = frame[t] if is_dynamic(args.dataset) else frame
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss
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
                            size=len(test_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples

        
        total_time = time.time() - start_time
        if (not multigpu or dist.get_rank()==0):
            print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
