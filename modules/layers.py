from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import surrogate
from .neurons import OnlineIFNode, OnlineLIFNode
import config
import copy

import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from datetime import datetime
import torch.distributed as dist

if torch.__version__ < "1.11.0":
    cpp_wrapper = load(name="cpp_wrapper", sources=["modules/cpp_wrapper.cpp"], verbose=True)
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_input(input.shape, grad_output, weight, padding, stride, dilation, groups,
                                                     cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_output, input, padding, stride, dilation, groups,
                                                      cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
else:
    bias_sizes, output_padding = [0, 0, 0, 0], [0, 0]
    transposed = False
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation,
                                            transposed, output_padding, groups, [True, False, False])[0]
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation,
                                            transposed, output_padding, groups, [False, True, False])[1]


def get_weight_sws(weight, gain, eps):
    fan_in = np.prod(weight.shape[1:])
    mean = torch.mean(weight, axis=[1, 2, 3], keepdims=True)
    var = torch.var(weight, axis=[1, 2, 3], keepdims=True)
    weight = (weight - mean) / ((var * fan_in + eps) ** 0.5)
    if gain is not None:
        weight = weight * gain
    return weight


class ScaledWSLinear(nn.Conv2d):

    def __init__(self, in_features, out_features, bias=True, gain=True):
        super(ScaledWSLinear, self).__init__(in_features, out_features, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1)) if gain else None
        self.eps = config.args.eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps) if config.args.WS else self.weight
        return F.Linear(x, weight, self.bias)


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = config.args.eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps) if config.args.WS else self.weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SynapseNeuron(nn.Module):
    def __init__(self, synapse=None, neuron_class=OnlineLIFNode, **kwargs):
        super().__init__()
        self.synapse = synapse
        self.init = False
        if isinstance(synapse, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.type = 'conv'
            shape = [1, synapse.out_channels, 1, 1]
        elif isinstance(synapse, nn.Linear):
            self.type = 'linear'
            shape = [1, synapse.out_features]
        else:
            raise NotImplementedError(f'Synapse type {type(synapse)} not supported!')

        if config.args.WS:
            self.gain = nn.Parameter(torch.ones(*shape)).transpose(0,1).cuda()
            self.eps = config.args.eps

        if config.args.BN:
            self.bn = MySyncBN(num_features=shape[1])
            # self.bn = nn.SyncBatchNorm(num_features=shape[1], momentum=0.1/config.args.T)

        if neuron_class == OnlineLIFNode:
            self.neuron = neuron_class(**kwargs)
        else:
            raise TypeError(f'Type of neuron can only be Online LIF Node! Current neuron type is {neuron_class}.')

    def forward(self, spike, **kwargs):
        init = kwargs.get('init', False)
        syn = self.synapse
        if init:
            self.init = True
            if self.type == 'conv':
                shape = list(spike.shape)
                shape[-3] = syn.out_channels
                shape[-2] = (shape[-2]+2*syn.padding[0]-syn.dilation[0]*(syn.kernel_size[0]-1)-1)//syn.stride[0]+1
                shape[-1] = (shape[-1]+2*syn.padding[1]-syn.dilation[1]*(syn.kernel_size[1]-1)-1)//syn.stride[1]+1
            else:
                shape = list(spike.shape)
                shape[-1] = syn.out_features

            self.neuron.forward_init(spike, shape=shape)

        # weight = get_weight_sws(syn.weight, self.gain, self.eps) if config.args.WS else syn.weight

        self.neuron.get_decay_coef()
        x = self.synapse(spike)
        if config.args.BN:
            x = self.bn(x, **kwargs)
            # x = self.bn(x)
        spike = self.neuron(x)
        self.init = False
        return spike


class MySyncBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('run_mean', torch.zeros(num_features))
        self.register_buffer('run_var', torch.ones(num_features))
        # for estimating total mean and var
        self.total_mean = 0.
        self.total_var = 0.
        self.momentum = 0.9

        self.last_training = False

    def forward(self, x, **kwargs):
        self.init = kwargs.get('init', False)
        return BNFunc.apply(x, self.gamma, self.beta, self)


class BNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, layer):
        # print(dist.get_rank(), weight.shape, torch.mean(weight), torch.var(weight))
        eps = config.args.eps
        if (layer.training and layer.init or not layer.training) and isinstance(layer.total_var, torch.Tensor):
            T = config.args.T_train if layer.training else config.args.T
            mean = layer.total_mean / T
            var = layer.total_var / T
            if config.args.BN_type == 'new': var -= mean ** 2
            layer.run_mean += (1 - layer.momentum) * (mean - layer.run_mean)
            layer.run_var += (1 - layer.momentum) * (var - layer.run_var)
            layer.total_mean = 0.
            layer.total_var = 0.

        mean, var, count_all = None, None, None
        if layer.training:
            # BN sync, refer to torch.nn.modules.batchnorm.SyncBatchNorm and torch.nn.modules._function.SyncBatchNorm
            need_sync = dist.is_available() and dist.is_initialized()
            if need_sync:
                process_group = dist.group.WORLD
                if dist.get_world_size(process_group) > 1:
                    mean, invstd, count_all = get_norm_stat_ddp(x, layer, process_group, eps)
                    var = (1. / invstd) ** 2 - eps
            if mean is None:
                mean, invstd = torch.batch_norm_stats(x, eps)
                var = (1. / invstd) ** 2 - eps
        else:
            mean, invstd = layer.run_mean, 1. / torch.sqrt(layer.run_var + eps)
        if count_all is None:
            count_all = torch.full((1,), x.numel() // x.size(1), dtype = mean.dtype, device=mean.device)

        if layer.training:
            layer.total_mean += mean
            layer.total_var += var
            if config.args.BN_type == 'new': layer.total_var += mean ** 2

        ctx.layer = layer
        ctx.save_for_backward(x, gamma, mean, invstd, count_all.to(torch.int32))
        if layer.training and config.args.BN_type == 'new':
            if torch.abs(torch.mean((1. / invstd)**2 - layer.run_var)) < 0.5:
                mean, invstd = layer.run_mean, 1. / torch.sqrt(layer.run_var + eps)
        x = torch.batch_norm_elemt(x, gamma, beta, mean, invstd, eps)
        return x

    @staticmethod
    def backward(ctx, grad):
        # shape of grad: B*C*H*W
        (x, gamma, mean, invstd, count_tensor) = ctx.saved_tensors
        if config.args.BN_type == 'new':
            gamma = gamma / (invstd * torch.sqrt(ctx.layer.run_var + config.args.eps))
        sum_dy, sum_dy_xmu, grad_gamma, grad_beta = torch.batch_norm_backward_reduce(grad, x, mean, invstd, gamma, True, True, True)

        # synchronizing stats used to calculate input gradient.
        if dist.is_available() and dist.is_initialized():
            process_group = dist.group.WORLD
            if dist.get_world_size(process_group) > 1:
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

        # backward pass for gradient calculation
        if gamma is not None and gamma.dtype != mean.dtype:
            gamma = gamma.to(mean.dtype)
        grad_x = torch.batch_norm_backward_elemt(grad, x, mean, invstd, gamma, sum_dy, sum_dy_xmu, count_tensor)

        return grad_x, grad_gamma, grad_beta, None, None


def get_norm_stat_ddp(input, layer, process_group, eps):
    world_size = dist.get_world_size(process_group)

    num_channels = input.shape[1]
    if input.numel() > 0:
        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        count = torch.full(
            (1,),
            input.numel() // input.size(1),
            dtype=mean.dtype,
            device=mean.device
        )

        # C, C, 1 -> (2C + 1)
        combined = torch.cat([mean, invstd, count], dim=0)
    else:
        # for empty input, set stats and the count to zero. The stats with
        # zero count will be filtered out later when computing global mean
        # & invstd, but they still needs to participate the all_gather
        # collective communication to unblock other peer processes.
        combined = torch.zeros(
            2 * num_channels + 1,
            dtype=input.dtype,
            device=input.device
        )

    if process_group._get_backend_name() != "gloo":
        # world_size * (2C + 1)
        combined_size = combined.numel()
        combined_flat = torch.empty(1,
                                    combined_size * world_size,
                                    dtype=combined.dtype,
                                    device=combined.device)
        dist.all_gather_into_tensor(combined_flat, combined, process_group, async_op=False)
        combined = torch.reshape(combined_flat, (world_size, combined_size))
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
    else:
        # world_size * (2C + 1)
        combined_list = [
            torch.empty_like(combined) for _ in range(world_size)
        ]
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

    if not torch.cuda.is_current_stream_capturing():
        # remove stats from empty inputs
        mask = count_all.squeeze(-1) >= 1
        count_all = count_all[mask]
        mean_all = mean_all[mask]
        invstd_all = invstd_all[mask]

    # calculate global mean & invstd
    counts = count_all.view(-1)
    running_mean, running_var, momentum = layer.run_mean, layer.run_var, layer.momentum
    if running_mean is not None and counts.dtype != running_mean.dtype:
        counts = counts.to(running_mean.dtype)
    mean, invstd = torch.batch_norm_gather_stats_with_counts(
        input, mean_all, invstd_all, None, None, momentum, eps, counts,
    )

    return mean, invstd, count_all
