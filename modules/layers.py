from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import surrogate
from .neurons import OnlineIFNode, OnlineLIFNode
import config
import math

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


@torch.jit.script
def get_mul(decay, s_out, v, dsdu):
    with torch.no_grad():
        # res = decay * (1 - s_out - v * dsdu)
        res = torch.tensor(decay)
    return res


def neuron_forward(layer, x, gamma, beta):
    neuron = layer.neuron
    v_last = neuron.v
    neuron.v_float_to_tensor(x)
    neuron.neuronal_charge(x)
    neuron.adjust_th() # newly added
    
    # unnormed_v = neuron.v
    # if config.args.BN:
    #     if layer.training:
    #         neuron.v, mean, var = bn_forward(neuron.v, gamma, beta, layer)
    #     else:
    #         neuron.v = (neuron.v - layer.run_mean) / torch.sqrt(layer.run_var + config.args.eps)
    #         neuron.v = neuron.v * gamma + beta
    #         mean, var = None, None
    #     a1 = gamma / torch.sqrt(layer.run_var + config.args.eps)
    #     a0 = beta - layer.run_mean * a1
    
    s_out = neuron.neuronal_fire()
    dsdu = neuron.surrogate_function.backward(torch.ones_like(s_out), neuron.v - neuron.v_threshold, neuron.surrogate_function.alpha)
    
    
    # subtraction reset may be too strong
    # neuron.v = unnormed_v
    if 0:#config.args.BN:
        neuron.v = neuron.v - s_out * (neuron.v_threshold - a0) / a1
    else:
        neuron.neuronal_reset(s_out)

    if neuron.dropout > 0.0 and neuron.training:
        s_out = neuron.mask.expand_as(s_out) * s_out
    neuron.spike = s_out
    # return s_out, dsdu, unnormed_v, mean, var
    return s_out, dsdu


@torch.jit.script
def get_inputs(s_in, s_in_acc, decay, v, s_out, dsdu, lvl):
    lvl = lvl.item()
    if lvl == 1:
        mul = torch.tensor(0.)
    elif lvl == 2:
        mul = torch.mean(decay)
    elif lvl == 3:
        mul = get_mul(decay, s_out, v, dsdu)
        mul = torch.mean(mul)
    elif lvl == 4:
        mul = get_mul(decay, s_out, v, dsdu)
        dims = [0] if len(mul.shape) <= 2 else [0,2,3]
        mul = torch.mean(mul, dim=dims, keepdim=True)
    else:
        mul = torch.tensor(0.)
        raise ValueError('Online level of weight out of range! (range: 1~4 integer)')
    
    s_in_acc *= mul
    s_in_acc += s_in
    return s_in_acc


def calc_grad_w(grad_w_func, grad_u, s_in, s_in_acc, decay, v, s_out, dsdu, lvl):
    inputs = get_inputs(s_in, s_in_acc, decay, v, s_out, dsdu, torch.tensor(lvl))
    # might change code here??
    grad_weight = grad_w_func(grad_u, inputs)
    return grad_weight


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
            self.gamma = nn.Parameter(torch.ones(*shape))
            self.beta = nn.Parameter(torch.zeros(*shape))
            self.run_mean = nn.Parameter(torch.zeros(*shape), requires_grad=False)
            self.run_var = nn.Parameter(torch.ones(*shape), requires_grad=False)
            self.count = 0
            self.last_training = False
            self.mul_acc = torch.ones(*shape).cuda()

            # for estimating total mean and var
            self.total_mean = torch.zeros(*shape).cuda()
            self.total_var = torch.zeros(*shape).cuda()
        else:
            self.gamma, self.beta = None, None

        if neuron_class == OnlineLIFNode:
            self.neuron = neuron_class(**kwargs)
        else:
            raise TypeError(f'Type of neuron can only be Online LIF Node! Current neuron type is {neuron_class}.')

    def forward(self, spike, **kwargs):
        if config.args.BN and self.training != self.last_training:
            with torch.no_grad():
                if self.training:
                    rate = 1/2 * (1 + math.cos(math.pi + math.pi * self.count / config.args.epochs))
                    # self.momentum = 0.8 + (0.95 - 0.8) * rate
                    self.momentum = 0.95
                    self.count += 1
                self.last_training = self.training
        
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
            self.s_in_acc = torch.zeros_like(spike, requires_grad=False)
            if config.args.BN:
                self.mul_acc = torch.ones_like(self.mul_acc)

        weight = get_weight_sws(syn.weight, self.gain, self.eps) if config.args.WS else syn.weight
        
        self.neuron.get_decay_coef()
        if config.args.weight_online_level == 0:
            pass
        else:
            if self.type == 'conv':
                spike = OnlineFunc.apply(spike, weight, syn.bias, self.gamma, self.beta, (syn.stride, syn.padding, syn.dilation, syn.groups), self)
            else:
                spike = OnlineFunc.apply(spike, weight, syn.bias, self.gamma, self.beta, None, self)
        self.init = False
        return spike


class OnlineFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_in, weight, bias, gamma, beta, convConfig, layer):
        # print(dist.get_rank(), weight.shape, torch.mean(weight), torch.var(weight))
        # need du/du (decay), du/ds (reset) and ds/du (surrogate)
        if layer.type == 'conv':
            x = F.conv2d(s_in, weight, bias, *convConfig)
            ctx.convConfig = convConfig
        else:
            x = F.linear(s_in, weight, bias)
        
        if config.args.BN:
            unnormed_x = x
            x, mean, var = bn_forward(x, gamma, beta, layer)
        s_out, dsdu = neuron_forward(layer, x, gamma, beta)
        # s_out, dsdu, unnormed_x, mean, var = neuron_forward(layer, x, gamma, beta)

        ctx.layer = layer
        if config.args.BN and layer.training:
            ctx.save_for_backward(s_in, weight, s_out, dsdu, unnormed_x, gamma, mean, var)
        else:
            ctx.save_for_backward(s_in, weight, s_out, dsdu)
        return s_out

    @staticmethod
    def backward(ctx, grad):
        # shape of grad: B*C*H*W
        layer = ctx.layer
        neuron = layer.neuron
        if config.args.BN:
            (s_in, weight, s_out, dsdu, x, gamma, mean, var) = ctx.saved_tensors
        else:
            (s_in, weight, s_out, dsdu) = ctx.saved_tensors

        grad_u = grad * dsdu

        if config.args.BN:
            var1 = layer.run_var if config.args.BN_type == 'new' else var
            grad_w_, grad_I, grad_gamma, grad_beta = bn_backward(grad_u, x, gamma, mean, var, var1, torch.tensor(config.args.eps))
        else:
            grad_w_, grad_I, grad_gamma, grad_beta = grad_u, grad_u, None, None
        grad_b = torch.sum(grad_I, dim=[i for i in range(len(grad_u.shape)) if i != 1], keepdim=False)

        if layer.type == 'conv':
            stride, padding, dilation, groups = ctx.convConfig
            grad_input = conv_backward_input(grad_I, s_in, weight, padding, stride, dilation, groups)
            grad_w_func = lambda grad_output, input: conv_backward_weight(grad_output, input, weight, padding, stride, dilation, groups)
        else:
            grad_input = torch.matmul(grad_I, weight)
            grad_w_func = lambda grad_output, input: torch.matmul(grad_output.transpose(1,2), input)
        # grad_weight = calc_grad_w(grad_w_func, grad_I, s_in, layer.s_in_acc, neuron.decay, neuron.v, s_out, dsdu, config.args.weight_online_level)
        grad_weight = calc_grad_w(grad_w_func, grad_w_, s_in, layer.s_in_acc, neuron.decay, neuron.v, s_out, dsdu, config.args.weight_online_level)

        if config.args.BN and config.args.weight_online_level >= 2:
            align_scale(layer.mul_acc, get_mul(neuron.decay, None, None, None), grad_weight)

        return grad_input, grad_weight, grad_b, grad_gamma, grad_beta, None, None


@torch.jit.script
def align_scale(mul_acc, decay, grad_weight):
    mul_acc *= decay
    mul_acc += 1
    grad_weight /= mul_acc.transpose(0,1)
    # grad_b *= layer.mul_acc
    # grad_gamma *= layer.mul_acc
    # grad_beta *= layer.mul_acc


def bn_forward(x, gamma, beta, layer):
    dims = [0] if len(x.shape) == 2 else [0,2,3]
    if layer.training:
        if layer.init:
            T = config.args.T_train if layer.training else config.args.T
            mean = layer.total_mean / T
            var = layer.total_var / T
            if config.args.BN_type == 'new': var -= mean ** 2
            layer.run_mean += (1 - layer.momentum) * (mean - layer.run_mean)
            layer.run_var += (1 - layer.momentum) * (var - layer.run_var)
            layer.total_mean = 0.
            layer.total_var = 0.

        mean, var = None, None
        # BN sync, refer to torch.nn.modules.batchnorm.SyncBatchNorm and torch.nn.modules._function.SyncBatchNorm
        need_sync = dist.is_available() and dist.is_initialized()
        if need_sync:
            process_group = dist.group.WORLD
            if dist.get_world_size(process_group) > 1:
                mean, invstd = get_norm_stat_ddp(x, layer, process_group, config.args.eps)
                bn_num = x.numel() / x.shape[1]
                var = ((1. / invstd) ** 2 - config.args.eps) * (1 - 1. / bn_num)
                shape = [1 for _ in x.shape]
                shape[1] = x.shape[1]
                mean, var = mean.reshape(shape), var.reshape(shape)
        
        if mean is None:
            mean = torch.mean(x, dim=dims, keepdim=True)
            var = torch.mean((x-mean)**2, dim=dims, keepdim=True)
        
        layer.total_mean += mean
        layer.total_var += var
        if config.args.BN_type == 'new': layer.total_var += mean ** 2

    if layer.training:
        if config.args.BN_type == 'old':
            x = calc_bn(x, mean, var, torch.tensor(config.args.eps), gamma, beta)
        else:
            x = calc_bn(x, layer.run_mean, layer.run_var, torch.tensor(config.args.eps), gamma, beta)
    else:
        x = calc_bn(x, layer.run_mean, layer.run_var, torch.tensor(config.args.eps), gamma, beta)
        mean, var = None, None
    return x, mean, var


@torch.jit.script
def calc_bn(x, mean, var, eps, gamma, beta):
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


@torch.jit.script
def bn_backward(grad_x, x, gamma, mean, var, var1, eps):
    gamma = gamma * var / var1

    dims = [0] if len(x.shape) == 2 else [0,2,3]
    std_inv = 1 / torch.sqrt(var + eps)
    x = (x - mean) * std_inv
    grad_beta = torch.sum(grad_x, dim=dims, keepdim=True)
    grad_gamma = torch.sum((grad_x * x), dim=dims, keepdim=True)
    grad_x = grad_x * gamma * std_inv
    m = x.numel() // x.shape[1]
    
    grad_w_ = grad_x - torch.sum(grad_x, dim=dims, keepdim=True) / m - torch.sum(grad_x * x, dim=dims, keepdim=True) * x / m
    grad_x = grad_w_
    return grad_w_, grad_x, grad_gamma, grad_beta


def get_norm_stat_ddp(input, layer, process_group, eps):
    world_size = dist.get_world_size(process_group)
    mean, invstd = torch.batch_norm_stats(input, eps)

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

    if process_group._get_backend_name() == 'nccl':
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
        # input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts,
        input, mean_all, invstd_all, None, None, momentum, eps, counts,
    )

    return mean, invstd