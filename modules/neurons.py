from typing import Callable, overload
import torch
import torch.nn as nn
from . import surrogate
from .neuron_spikingjelly import IFNode, LIFNode, ParametricLIFNode
import config
import math

class OnlineIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = None,
            surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = True,
            neuron_dropout: float = 0.0, **kwargs):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.dropout = neuron_dropout
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)
    
    def get_decay_coef(self):
        self.decay = 0

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        return spike


class OnlineLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.dropout = neuron_dropout
        if self.dropout > 0.0:
            self.register_memory('mask', None)
        self.init = None
        self.init_threshold = v_threshold

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * self.decay + x
        else:
            self.v = self.v.detach() * self.decay + self.v_reset * (1. - self.decay) + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor, shape=None):
        if shape is None:
            self.v = torch.zeros_like(x)
        else:
            self.v = torch.zeros(*shape, device=x.device)
        # self.v = 0.
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)
        self.init = True
    
    def get_decay_coef(self):
        self.decay = torch.tensor(1 - 1. / self.tau)
    
    def adjust_th(self):
        if config.args.dynamic_threshold:
            with torch.no_grad():
                x = self.v
                mean, std = torch.mean(x), torch.std(x)
                if self.init:
                    self.th_ratio = (self.init_threshold - mean) / std
                    self.init = False
                self.v_threshold = mean + std * self.th_ratio

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        if init:
            self.forward_init(x)

        self.get_decay_coef()
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.adjust_th() # newly added
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        self.spike = spike
        return spike


class MyLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.spike = None
    
    def single_step_forward(self, x: torch.Tensor, **kwargs):
        spike = super().single_step_forward(x)
        self.spike = spike
        return spike


class OnlinePLIFNode(ParametricLIFNode):
    def __init__(self, tau: float = 2., tau_shape = [1], decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(tau - 1.)
        self.w = nn.Parameter(torch.ones(*tau_shape) * init_w)

        self.dropout = neuron_dropout
        self.spike = None
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0.:
            self.v = self.v.detach() * self.decay + x
        else:
            self.v = self.v.detach() * self.decay + self.v_reset * (1. - self.decay) + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor, shape=None):
        # x: B * C * H * W
        if shape is None:
            shape = x.shape
        self.v = torch.zeros(*shape, device=x.device)
        
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)
        self.decay_acc = torch.zeros(*shape, device=x.device, requires_grad=False)
    
    def get_decay_coef(self):
        self.decay = self.w.sigmoid()

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        if init:
            self.forward_init(x)

        self.get_decay_coef()
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        return spike