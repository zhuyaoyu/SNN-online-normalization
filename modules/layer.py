import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import surrogate
from .neuron import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode

import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from datetime import datetime

cpp_wrapper = load(name="cpp_wrapper", sources=["modules/cpp_wrapper.cpp"], verbose=True)


def get_weight_sws(weight, gain, eps):
    fan_in = np.prod(weight.shape[1:])
    mean = torch.mean(weight, axis=[1, 2, 3], keepdims=True)
    var = torch.var(weight, axis=[1, 2, 3], keepdims=True)
    weight = (weight - mean) / ((var * fan_in + eps) ** 0.5)
    if gain is not None:
        weight = weight * gain
    return weight


def neuron_forward(neuron, s_in, x):
    v_last = neuron.v
    neuron.neuronal_charge(x)
    s_out = neuron.neuronal_fire()
    dudu = neuron.decay
    duds = s_out * neuron.v_threshold if neuron.v_reset is None else s_out * (neuron.v - neuron.v_reset)
    dsdu = neuron.surrogate_function.backward(torch.ones_like(s_out), neuron.v - neuron.v_threshold, neuron.surrogate_function.alpha)
    if isinstance(neuron, OnlinePLIFNode):
        neuron.decay_acc = v_last + neuron.decay_acc * neuron.mul
        neuron.mul = neuron.decay * (1 - s_out - neuron.v * dsdu)
        # neuron.mul = 0
    
    neuron.neuronal_reset(s_out)
    return s_out, dsdu


class ScaledWSLinear(nn.Conv2d):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4):
        super(ScaledWSLinear, self).__init__(in_features, out_features, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        return F.Linear(x, weight, self.bias)


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps

    def forward(self, x, **kwargs):
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SWSLinearNeuron(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4, neuron=OnlineLIFNode(), **kwargs):
        super(SWSLinearNeuron, self).__init__(in_features, out_features, bias)
        self.gain = nn.Parameter(torch.ones(self.out_features, 1)) if gain else None
        self.eps = eps
        self.neuron = OnlineLIFNode(tau=tau, decay_input=False, surrogate_function=surrogate_function, dropout=dropout)

    def forward(self, x, **kwargs):
        init = kwargs.get('init', False)
        if init:
            # shape = list(spike.shape)
            shape[-1] = self.out_features
            self.neuron.forward_init(spike, shape=shape)
            self.w_acc = torch.zeros_like(self.weight)
        weight = get_weight_sws(self.weight, self.gain, self.eps)

        self.neuron.get_decay_coef()
        spike = OnlineLinear.apply(spike, weight, self.bias, self.decay, self)
        return spike


class OnlineLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_in, weight, bias, decay, layer):
        # need du/du (decay), du/ds (reset) and ds/du (surrogate)
        x = F.linear(s_in, weight, bias)
        s_out, dsdu = neuron_forward(layer.neuron, s_in, x)

        ctx.save_for_backward(s_in, weight, dsdu)
        ctx.layer = layer
        return s_out
    
    @staticmethod
    def backward(ctx, grad):
        layer = ctx.layer
        (s_in, weight, dsdu) = ctx.saved_tensors
        grad_u = grad * dsdu
        
        grad_input = torch.matmul(grad_u, weight)
        grad_weight = torch.sum(torch.matmul(grad_u.transpose(1,2), s_in), dim=0)
        layer.w_acc = layer.w_acc * layer.neuron.decay + grad_weight
        # grad_weight = layer.w_acc
        # u = u * decay + conv(s_in, weight) + bias, d(u)/d(bias) = 1
        
        grad_b = torch.sum(grad_u, dim=[i for i in range(len(grad_u.shape)) if i != 1])
        if isinstance(layer.neuron, OnlinePLIFNode):
            grad_decay = torch.mean(grad_u * neuron.decay_acc, dim=[0], keepdim=True)
        else:
            grad_decay = None
        
        return grad_input, grad_weight, grad_b, grad_decay, None


class SWSConvNeuron(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4, neuron=OnlineLIFNode(), **kwargs):
        super(SWSConvNeuron, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps
        self.neuron = neuron

    def forward(self, spike, **kwargs):
        init = kwargs.get('init', False)
        if init:
            shape = list(spike.shape)
            shape[-3] = self.out_channels
            shape[-2] = (shape[-2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
            shape[-1] = (shape[-1]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1
            self.neuron.forward_init(spike, shape=shape)
            self.w_acc = torch.zeros_like(self.weight)
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        
        self.neuron.get_decay_coef()
        spike = OnlineConv.apply(spike, weight, self.bias, self.neuron.decay, (self.stride, self.padding, self.dilation, self.groups), self)
        return spike


class OnlineConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_in, weight, bias, decay, convConfig, layer):
        # need du/du (decay), du/ds (reset) and ds/du (surrogate)
        x = F.conv2d(s_in, weight, bias, *convConfig)
        s_out, dsdu = neuron_forward(layer.neuron, s_in, x)

        ctx.layer = layer
        ctx.convConfig = convConfig
        ctx.save_for_backward(s_in, weight, dsdu)
        return s_out
    
    @staticmethod
    def backward(ctx, grad):
        layer = ctx.layer
        stride, padding, dilation, groups = ctx.convConfig
        (s_in, weight, dsdu) = ctx.saved_tensors
        grad_u = grad * dsdu
        grad_input = cpp_wrapper.cudnn_convolution_backward_input(s_in.shape, grad_u, weight, padding,
                                                                  stride, dilation, groups,
                                                                  cudnn.benchmark, cudnn.deterministic,
                                                                  cudnn.allow_tf32)
        grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_u, s_in, padding,
                                                                    stride, dilation, groups,
                                                                    cudnn.benchmark, cudnn.deterministic,
                                                                    cudnn.allow_tf32)
        layer.w_acc = layer.w_acc * torch.mean(layer.neuron.decay) + grad_weight
        # grad_weight = layer.w_acc
        # u = u * decay + conv(s_in, weight) + bias, d(u)/d(bias) = 1
        grad_b = torch.sum(grad_u, dim=[i for i in range(len(grad_u.shape)) if i != 1])
        if isinstance(layer.neuron, OnlinePLIFNode):
            grad_decay = torch.mean(grad_u * layer.neuron.decay_acc, dim=[0,2,3], keepdim=True)
        else:
            grad_decay = None
        return grad_input, grad_weight, grad_b, grad_decay, None, None