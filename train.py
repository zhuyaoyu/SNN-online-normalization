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
from models import spiking_vgg, spiking_resnet_imagenet
from modules import layers, neurons, surrogate, neuron_spikingjelly
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets.data import get_dataset
import config
import argparse
import math
import torch.utils.data as data
import numpy as np

import random


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
    parser.add_argument('-dist', type=str, default="nccl", help='distributed data parallel backend')
    parser.add_argument('--local-rank', type=int, default=-1)
    
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
        net = spiking_vgg.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, neuron_dropout=args.drop_rate, fc_hw=1, BN=args.BN, weight_standardization=args.WS)
    else:
        neuron0 = neurons.OnlineLIFNode if not args.BPTT else neurons.MyLIFNode
        net = spiking_resnet_imagenet.__dict__[args.model](single_step_neuron=neuron0, tau=args.tau, surrogate_function=surrogate.Sigmoid(), c_in=c_in, num_classes=num_classes, drop_rate=args.drop_rate, stochdepth_rate=args.stochdepth_rate, neuron_dropout=0.0, zero_init_residual=False)
    #print(net)
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()
    if multigpu:
        net = DDP(net)

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

    out_dir = os.path.join(args.out_dir, f'{args.model}_{args.cnf}_T_{args.T}_T_train_{args.T_train}_{args.opt}_lr_{args.lr}_tau_{args.tau}_taulvl_{args.tau_online_level}_wlvl_{args.weight_online_level}_')
    if args.BPTT:
        out_dir += 'BPTT_'
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)
        #assert args.resume is not None

    # pt_dir = out_dir + '_pt'
    pt_dir = out_dir
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')


    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)
    
    criterion_mse = nn.MSELoss(reduce=True)

    if multigpu:
        init_seeds(1 + cfg.local_rank)
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        if multigpu:
            train_loader.sampler.set_epoch(epoch)
        (net.module if multigpu else net).train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        if (not multigpu or dist.get_rank()==0):
            bar = Bar('Processing', max=len(train_loader))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0
        for frame, label in train_loader:
            batch_idx += 1
            frame = frame.float().cuda()
            t_step = args.T_train if args.T_train is not None else args.T

            if args.T_train and args.T_train != args.T:
                assert(is_dynamic(args.dataset))
                sec_list = np.random.choice(frame.shape[1], args.T_train, replace=False)
                sec_list.sort()
                frame = frame[:, sec_list]
                t_step = args.T_train

            label = label.cuda()

            batch_loss = 0
            if args.BPTT:
                bptt_loss = 0
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            for t in range(t_step):
                input_frame = frame[:, t] if is_dynamic(args.dataset) else frame
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
            if (not multigpu or dist.get_rank()==0):
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

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        (net.module if multigpu else net).eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        if (not multigpu or dist.get_rank()==0):
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

                for t in range(t_step):
                    input_frame = frame[:, t] if is_dynamic(args.dataset) else frame
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
                if (not multigpu or dist.get_rank()==0):
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
        if (not multigpu or dist.get_rank()==0):
            bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': (net.module if multigpu else net).state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))
        #for item in sys.argv:
        #    print(item, end=' ')
        #print('')
        #print(args)
        #print(out_dir)
        total_time = time.time() - start_time
        if (not multigpu or dist.get_rank()==0):
            print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
