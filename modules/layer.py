import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_weight_sws(weight, gain, eps):
    fan_in = np.prod(weight.shape[1:])
    mean = torch.mean(weight, axis=[1, 2, 3], keepdims=True)
    var = torch.var(weight, axis=[1, 2, 3], keepdims=True)
    weight = (weight - mean) / ((var * fan_in + eps) ** 0.5)
    if gain is not None:
        weight = weight * gain


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def forward(self, x):
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledWSLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4):
        super(ScaledWSLinear, self).__init__(in_features, out_features, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        else:
            self.gain = None
        self.eps = eps

    def forward(self, x):
        weight = get_weight_sws(self.weight, self.gain, self.eps)
        return F.linear(x, weight, self.bias)