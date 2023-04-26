import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import surrogate
from .neuron import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode
import config

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
        res = decay * (1 - s_out - v * dsdu)
    return res


def neuron_forward(neuron, s_in, x):
    v_last = neuron.v
    neuron.neuronal_charge(x)
    s_out = neuron.neuronal_fire()
    # dudu = neuron.decay
    # duds = s_out * neuron.v_threshold if neuron.v_reset is None else s_out * (neuron.v - neuron.v_reset)
    if config.args.tau_online_level >= 3:
        dsdu = neuron.surrogate_function.backward(torch.ones_like(s_out), neuron.v - neuron.v_threshold, neuron.surrogate_function.alpha)
        mul = get_mul(neuron.decay, s_out, neuron.v, dsdu)
    if isinstance(neuron, OnlinePLIFNode):
        lvl = config.args.tau_online_level
        # print('config.args.tau_online_level =', lvl)
        if lvl == 1:
            neuron.decay_acc = v_last
        elif lvl == 2:
            neuron.decay_acc = v_last + neuron.decay_acc * neuron.decay
        elif lvl == 3:
            neuron.decay_acc = v_last + neuron.decay_acc * torch.mean(mul)
        elif lvl == 4:
            dim = [i for i in range(len(mul.shape)) if i != 0]
            neuron.decay_acc = v_last + neuron.decay_acc * torch.mean(mul, dim=dim, keepdim=True)
        elif lvl == 5:
            neuron.decay_acc = v_last + neuron.decay_acc * mul
        else:
            raise ValueError('Online level of tau out of range! (range: 1~5 integer)')
    
    neuron.neuronal_reset(s_out)
    if neuron.dropout > 0.0 and neuron.training:
        s_out = neuron.mask.expand_as(s_out) * s_out
    neuron.spike = s_out
    return s_out


def calc_grad_w(grad_w_func, grad_u, s_in, layer, neuron, s_out, dsdu, lvl):
    if config.args.weight_online_level >= 3:
        mul = get_mul(neuron.decay, s_out, neuron.v, dsdu)
    if lvl == 1:
        grad_weight = grad_w_func(grad_u, s_in)
    else:
        if lvl == 2:
            mul = torch.mean(neuron.decay)
        elif lvl == 3:
            mul = torch.mean(mul)
        elif lvl == 4:
            dim = [i for i in range(len(mul.shape)) if i != 0]
            mul = torch.mean(mul, dim=dim, keepdim=True)
        else:
            raise ValueError('Online level of weight out of range! (range: 1~4 integer)')
        
        layer.s_in_acc = layer.s_in_acc * mul + s_in
        grad_weight = grad_w_func(grad_u, layer.s_in_acc)
    return grad_weight


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


class SWSConvNeuron(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4, neuron_class=OnlineLIFNode, **kwargs):
        super(SWSConvNeuron, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.eps = eps
        if neuron_class == OnlineLIFNode:
            self.neuron = neuron_class()
        elif neuron_class == OnlinePLIFNode:
            # self.neuron = neuron_class(tau_shape = (1, self.out_channels, 1, 1))
            self.neuron = neuron_class(tau_shape = (1,))
        else:
            raise TypeError('Type of neuron can only be Online LIF Node or Online PLIF Node!')

    def forward(self, spike, **kwargs):
        init = kwargs.get('init', False)
        if init:
            shape = list(spike.shape)
            shape[-3] = self.out_channels
            shape[-2] = (shape[-2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
            shape[-1] = (shape[-1]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1
            self.neuron.forward_init(spike, shape=shape)
            self.s_in_acc = torch.zeros_like(spike, requires_grad=False)
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        
        self.neuron.get_decay_coef()
        spike = OnlineConv.apply(spike, weight, self.bias, self.neuron.decay, (self.stride, self.padding, self.dilation, self.groups), self)
        return spike


class OnlineConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_in, weight, bias, decay, convConfig, layer):
        # need du/du (decay), du/ds (reset) and ds/du (surrogate)
        x = F.conv2d(s_in, weight, bias, *convConfig)
        s_out = neuron_forward(layer.neuron, s_in, x)

        ctx.layer = layer
        ctx.convConfig = convConfig
        ctx.save_for_backward(s_in, weight, s_out)
        return s_out
    
    @staticmethod
    def backward(ctx, grad):
        # shape of grad: B*C*H*W
        layer = ctx.layer
        neuron = layer.neuron
        stride, padding, dilation, groups = ctx.convConfig
        (s_in, weight, s_out) = ctx.saved_tensors
        dsdu = neuron.surrogate_function.backward(torch.ones_like(neuron.v), neuron.v - neuron.v_threshold, neuron.surrogate_function.alpha)
        grad_u = grad * dsdu
        grad_b = torch.sum(grad_u, dim=[i for i in range(len(grad_u.shape)) if i != 1])
        grad_input = conv_backward_input(grad_u, s_in, weight, padding, stride, dilation, groups)
        
        grad_w_func = lambda grad_output, input: conv_backward_weight(grad_output, input, weight, padding, stride, dilation, groups)
        grad_weight = calc_grad_w(grad_w_func, grad_u, s_in, layer, neuron, s_out, dsdu, config.args.weight_online_level)
        
        if isinstance(neuron, OnlinePLIFNode):
            # grad_decay = torch.sum(grad_u * neuron.decay_acc, dim=[0,2,3], keepdim=True)
            grad_decay = torch.sum(grad_u * neuron.decay_acc).reshape(1)
        else:
            grad_decay = None
        return grad_input, grad_weight, grad_b, grad_decay, None, None


class SWSLinearNeuron(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4, neuron_class=OnlineLIFNode, **kwargs):
        super(SWSLinearNeuron, self).__init__(in_features, out_features, bias)
        self.gain = nn.Parameter(torch.ones(self.out_features, 1)) if gain else None
        self.eps = eps
        if neuron_class == OnlineLIFNode:
            self.neuron = neuron_class()
        elif neuron_class == OnlinePLIFNode:
            # self.neuron = neuron_class(tau_shape = (1, self.out_channels))
            self.neuron = neuron_class(tau_shape = (1,))
        else:
            raise TypeError('Type of neuron can only be Online LIF Node or Online PLIF Node!')

    def forward(self, x, **kwargs):
        init = kwargs.get('init', False)
        if init:
            # shape = list(spike.shape)
            shape[-1] = self.out_features
            self.neuron.forward_init(spike, shape=shape)
            self.s_in_acc = torch.zeros_like(spike, requires_grad=False)
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
        neuron = layer.neuron
        (s_in, weight, dsdu) = ctx.saved_tensors
        grad_u = grad * dsdu
        grad_b = torch.sum(grad_u, dim=[i for i in range(len(grad_u.shape)) if i != 1])
        grad_input = torch.matmul(grad_u, weight)

        grad_w_func = lambda grad_output, input: torch.matmul(grad_output.transpose(1,2), input)
        grad_weight = calc_grad_w(grad_w_func, grad, grad_u, s_in, layer, neuron, config.args.weight_online_level)
        
        if isinstance(neuron, OnlinePLIFNode):
            # grad_decay = torch.sum(grad_u * neuron.decay_acc, dim=[0], keepdim=True)
            grad_decay = torch.sum(grad_u * neuron.decay_acc).reshape(1)
        else:
            grad_decay = None
        
        return grad_input, grad_weight, grad_b, grad_decay, None