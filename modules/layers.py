from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import surrogate
from .neurons import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode
import config
import math

import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from datetime import datetime

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
    
    # unnormed_v = neuron.v
    # if config.args.BN:
    #     if layer.training:
    #         neuron.v, mean, var = bn_forward(neuron.v, gamma, beta, layer)
    #     else:
    #         neuron.v = (neuron.v - layer.run_mean) / torch.sqrt(layer.run_var + 1e-4)
    #         neuron.v = neuron.v * gamma + beta
    #         mean, var = None, None
    #     a1 = gamma / torch.sqrt(layer.run_var + 1e-4)
    #     a0 = beta - layer.run_mean * a1
    
    s_out = neuron.neuronal_fire()
    dsdu = neuron.surrogate_function.backward(torch.ones_like(s_out), neuron.v - neuron.v_threshold, neuron.surrogate_function.alpha)
    
    lvl = config.args.tau_online_level
    if lvl >= 3:
        mul = get_mul(neuron.decay, s_out, neuron.v, dsdu)
    if isinstance(neuron, OnlinePLIFNode):
        # print('config.args.tau_online_level =', lvl)
        if lvl == 1:
            neuron.decay_acc = v_last
        elif lvl == 2:
            neuron.decay_acc = v_last + neuron.decay_acc * neuron.decay
        elif lvl == 3:
            neuron.decay_acc = v_last + neuron.decay_acc * torch.mean(mul)
        elif lvl == 4:
            dim = [0] if len(mul.shape) == 2 else [0,2,3]
            neuron.decay_acc = v_last + neuron.decay_acc * torch.mean(mul, dim=dim, keepdim=True)
        elif lvl == 5:
            neuron.decay_acc = v_last + neuron.decay_acc * mul
        else:
            raise ValueError('Online level of tau out of range! (range: 1~5 integer)')
    
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
    grad_weight = grad_w_func(grad_u, inputs)
    return grad_weight


class ScaledWSLinear(nn.Conv2d):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4):
        super(ScaledWSLinear, self).__init__(in_features, out_features, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps) if config.args.WS else self.weight
        return F.Linear(x, weight, self.bias)


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps) if config.args.WS else self.weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SynapseNeuron(nn.Module):
    def __init__(self, synapse=None, gain=True, eps=1e-4, neuron_class=OnlineLIFNode, **kwargs):
        super().__init__()
        self.synapse = synapse
        if isinstance(synapse, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.type = 'conv'
            shape = [1, synapse.out_channels, 1, 1]
        elif isinstance(synapse, nn.Linear):
            self.type = 'linear'
            shape = [1, synapse.out_features]
        else:
            raise NotImplementedError(f'Synapse type {type(synapse)} not supported!')
        
        self.gain = nn.Parameter(torch.ones(*shape).transpose(0,1)) if gain else None
        self.eps = eps
        if config.args.BN:
            self.gamma = nn.Parameter(torch.ones(*shape))
            self.beta = nn.Parameter(torch.zeros(*shape))
            self.run_mean = nn.Parameter(torch.zeros(*shape), requires_grad=False)
            self.run_var = nn.Parameter(torch.zeros(*shape), requires_grad=False)
            self.count = 0
            self.last_training = False
            self.mul_acc = torch.ones(*shape).cuda()

            # for estimating total mean and var
            self.total_mean = torch.ones(*shape).cuda()
            self.total_var = torch.ones(*shape).cuda()

        if neuron_class == OnlineLIFNode:
            self.neuron = neuron_class(**kwargs)
        elif neuron_class == OnlinePLIFNode:
            # self.neuron = neuron_class(tau_shape = (1, self.out_channels, 1, 1))
            self.neuron = neuron_class(tau_shape = (1,), **kwargs)
        else:
            raise TypeError('Type of neuron can only be Online LIF Node or Online PLIF Node!')

    def forward(self, spike, **kwargs):
        if self.training != self.last_training:
            with torch.no_grad():
                if self.training:
                    rate = 1/2 * (1 + math.cos(math.pi + math.pi * self.count / config.args.epochs))
                    self.momentum = 0.8 + (0.95 - 0.8) * rate
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
            self.mul_acc = torch.ones_like(self.mul_acc)

        weight = get_weight_sws(syn.weight, syn.gain, self.eps) if config.args.WS else syn.weight
        
        self.neuron.get_decay_coef()
        if self.type == 'conv':
            spike = OnlineFunc.apply('conv', spike, weight, syn.bias, self.neuron.decay, self.gamma, self.beta, (syn.stride, syn.padding, syn.dilation, syn.groups), self)
        else:
            spike = OnlineFunc.apply('linear', spike, weight, syn.bias, self.neuron.decay, self.gamma, self.beta, None, self)
        self.init = False
        return spike


class OnlineFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, type, s_in, weight, bias, decay, gamma, beta, convConfig, layer):
        # need du/du (decay), du/ds (reset) and ds/du (surrogate)
        if type == 'conv':
            x = F.conv2d(s_in, weight, bias, *convConfig)
            ctx.convConfig = convConfig
        else:
            x = F.linear(s_in, weight, bias)
        
        if config.args.BN:
            unnormed_x = x
            if layer.training:
                x, mean, var = bn_forward(x, gamma, beta, layer)
            else:
                x = (x - layer.run_mean) / torch.sqrt(layer.run_var + 1e-4)
                x = x * gamma + beta
                mean, var = None, None
        s_out, dsdu = neuron_forward(layer, x, gamma, beta)
        # s_out, dsdu, unnormed_x, mean, var = neuron_forward(layer, x, gamma, beta)

        ctx.layer = layer
        ctx.type = type
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
        if isinstance(neuron, OnlinePLIFNode):
            # grad_decay = torch.sum(grad_u * neuron.decay_acc, dim=[0,2,3], keepdim=True)
            grad_decay = torch.sum(grad_u * neuron.decay_acc).reshape(1)
        else:
            grad_decay = None

        if config.args.BN:
            grad_I, grad_gamma, grad_beta = bn_backward(grad_u, x, gamma, mean, var, layer.run_var)
        else:
            grad_I, grad_gamma, grad_beta = grad_u, None, None
        grad_b = torch.sum(grad_I, dim=[i for i in range(len(grad_u.shape)) if i != 1], keepdim=False)

        if ctx.type == 'conv':
            stride, padding, dilation, groups = ctx.convConfig
            grad_input = conv_backward_input(grad_I, s_in, weight, padding, stride, dilation, groups)
            grad_w_func = lambda grad_output, input: conv_backward_weight(grad_output, input, weight, padding, stride, dilation, groups)
        else:
            grad_input = torch.matmul(grad_I, weight)
            grad_w_func = lambda grad_output, input: torch.matmul(grad_output.transpose(1,2), input)
        grad_weight = calc_grad_w(grad_w_func, grad_I, s_in, layer.s_in_acc, neuron.decay, neuron.v, s_out, dsdu, config.args.weight_online_level)

        if config.args.weight_online_level >= 2:
            layer.mul_acc = 1 + layer.mul_acc * get_mul(neuron.decay, None, None, None)
            # grad_gamma *= layer.mul_acc
            # grad_beta *= layer.mul_acc
            grad_b *= layer.mul_acc.reshape(-1)

        return None, grad_input, grad_weight, grad_b, grad_decay, grad_gamma, grad_beta, None, None


# @torch.jit.script
def bn_forward(x, gamma, beta, layer):
    dims = [0] if len(x.shape) == 2 else [0,2,3]
    if layer.training:
        if layer.init:
            T = config.args.T_train if layer.training else config.args.T
            layer.run_mean += (1 - layer.momentum) * (layer.total_mean / T - layer.run_mean)
            layer.run_var += (1 - layer.momentum) * (layer.total_var / T - layer.run_var)
            layer.total_mean = 0.
            layer.total_var = 0.

        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.mean((x-layer.run_mean)**2, dim=dims, keepdim=True)
        # var = torch.mean((x-mean)**2, dim=dims, keepdim=True)

        if layer.init and torch.mean(layer.run_var) < torch.mean(var) / 2:
            layer.run_var += - layer.run_var + var

        layer.total_mean += mean
        layer.total_var += var

    x = (x - layer.run_mean) / torch.sqrt(layer.run_var + 1e-4) * gamma + beta
    # x = (x - mean) / torch.sqrt(var + 1e-4) * gamma + beta
    return x, mean, var


@torch.jit.script
def bn_backward(grad_x, x, gamma, mean, var, var1):
    gamma = gamma * var / var1

    dims = [0] if len(x.shape) == 2 else [0,2,3]
    std_inv = 1 / torch.sqrt(var + 1e-4)
    x = (x - mean) * std_inv
    grad_beta = torch.sum(grad_x, dim=dims, keepdim=True)
    grad_gamma = torch.sum((grad_x * x), dim=dims, keepdim=True)
    grad_x = grad_x * gamma * std_inv
    m = x.numel() // x.shape[1]
    grad_x = grad_x - (torch.sum(grad_x * x, dim=dims, keepdim=True) * x + torch.sum(grad_x, dim=dims, keepdim=True)) / m
    return grad_x, grad_gamma, grad_beta


class MyBN(nn.Module):
    def __init__(self, channels, eps=1e-4, **kwargs):
        super(MyBN, self).__init__(**kwargs)
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.run_mean = torch.zeros(channels, requires_grad=False)
        self.run_var = torch.zeros(channels, requires_grad=False)
        self.momentum = 0.99

    def forward(self, x, **kwargs):
        shape1 = [1 for _ in x.shape]
        shape1[1] = x.shape[1]
        gamma, beta = self.gamma.reshape(shape1), self.beta.reshape(shape1)
        self.run_mean, self.run_var = map(lambda a: a.to(x).reshape(shape1), (self.run_mean, self.run_var))
        if self.training:
            x, mean, var = bn_forward(x, gamma, beta, self.run_mean, self.run_var, torch.tensor(self.momentum))
        else:
            x = (x - self.run_mean) / torch.sqrt(self.run_var + 1e-4)
            x = x * gamma + beta
        return x
        # return BNFunc.apply(x, gamma, beta, self)


class BNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, layer):
        if layer.training:
            unnormed_x = x
            layer.run_mean, layer.run_var = layer.run_mean.to(x), layer.run_var.to(x)
            x, mean, var = bn_forward(x, gamma, beta, layer.run_mean, layer.run_var, torch.tensor(layer.momentum))
            ctx.save_for_backward(unnormed_x, gamma, mean, var)
        else:
            x = (x - layer.run_mean) / torch.sqrt(layer.run_var + 1e-4)
            x = x * gamma + beta
        return x
    
    @staticmethod
    def backward(ctx, grad):
        # shape of grad: B*C*H*W
        (x, gamma, mean, var) = ctx.saved_tensors
        grad_I, grad_gamma, grad_beta = bn_backward(grad, x, gamma, mean, var)

        return grad_I, grad_gamma, grad_beta, None
