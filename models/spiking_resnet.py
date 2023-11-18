import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from modules import neuron_spikingjelly
from modules.neurons import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, MyLIFNode
from modules.layers import ScaledWSConv2d, ScaledWSLinear, SynapseNeuron, MySyncBN
import torch.distributed as dist
import config

__all__ = ["online_spiking_resnet17", "online_spiking_resnet19"]

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class SequentialModule(nn.Sequential):
    def __init__(self, *args):
        super(SequentialModule, self).__init__(*args)

    def forward(self, input, **kwargs):
        for module in self._modules.values():
            if isinstance(module, (OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, MySyncBN, SynapseNeuron, BasicBlock)):
                input = module(input, **kwargs)
            else:
                if isinstance(module, neuron_spikingjelly.BaseNode):
                    input = module.single_step_forward(input)
                else:
                    input = module(input)
        return input


class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron: callable = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn1 = neuron(decay_input=False, v_reset=None, tau=kwargs.get("tau", 2.0))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn2 = neuron(decay_input=False, v_reset=None, tau=kwargs.get("tau", 2.0))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, **kwargs)
        out = self.sn1(out, **kwargs)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x, **kwargs)
        out = out + identity

        out = self.bn2(out, **kwargs)
        out = self.sn2(out, **kwargs)

        return out


class OnlineSpikingResNet17(nn.Module):
    def __init__(self, block=None, num_classes=10, norm_layer=None, neuron=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = MySyncBN
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neuron(decay_input=False, v_reset=None, tau=kwargs.get("tau", 2.0))

        self.layer1 = self._make_layer(block, 64, 3, stride=2, neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, 4, stride=2, neuron=neuron, **kwargs)
        self.avgpool = nn.AvgPool2d(2, 2)
        W = 32 // 2 ** 3
        self.fc1 = nn.Sequential(
            nn.Linear(128*W*W, 256),
            norm_layer(256),
        )
        self.fc2 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False, neuron: callable = None, **kwargs,) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SequentialModule(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, previous_dilation, norm_layer, neuron, **kwargs)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=self.dilation, norm_layer=norm_layer, neuron=neuron, **kwargs)
            )

        return SequentialModule(*layers)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x, **kwargs)
        x = self.sn1(x, **kwargs)

        x = self.layer1(x, **kwargs)
        x = self.layer2(x, **kwargs)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class OnlineSpikingResNet19(nn.Module):
    def __init__(self, block=None, num_classes=10, norm_layer=None, neuron=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = MySyncBN
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neuron(decay_input=False, v_reset=None, tau=kwargs.get("tau", 2.0))

        self.layer1 = self._make_layer(block, 128, 3, stride=1, neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 256, 3, stride=2, neuron=neuron, **kwargs)
        self.layer3 = self._make_layer(block, 512, 2, stride=2, neuron=neuron, **kwargs)
        self.avgpool = nn.AvgPool2d(2, 2)
        W = 32 // 2 ** 3
        self.fc1 = nn.Sequential(
            nn.Linear(512*W*W, 256),
            norm_layer(256),
        )
        self.fc2 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False, neuron: callable = None, **kwargs,) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SequentialModule(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, previous_dilation, norm_layer, neuron, **kwargs)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=self.dilation, norm_layer=norm_layer, neuron=neuron, **kwargs)
            )

        return SequentialModule(*layers)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x, **kwargs)
        x = self.sn1(x, **kwargs)

        x = self.layer1(x, **kwargs)
        x = self.layer2(x, **kwargs)
        x = self.layer3(x, **kwargs)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def online_spiking_resnet17(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return OnlineSpikingResNet17(block=BasicBlock, neuron=single_step_neuron, **kwargs)


def online_spiking_resnet19(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return OnlineSpikingResNet19(block=BasicBlock, neuron=single_step_neuron, **kwargs)


# def _online_spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
#     model = OnlineSpikingResNet(block, layers, neuron=single_step_neuron, **kwargs)
#     return model


# def online_spiking_resnet18(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
#     return _online_spiking_resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, single_step_neuron, **kwargs)
