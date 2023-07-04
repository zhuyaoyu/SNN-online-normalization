import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from modules import neuron_spikingjelly
from modules.neurons import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, MyLIFNode
from modules.layers import ScaledWSConv2d, ScaledWSLinear, SynapseNeuron
import config

__all__ = ['OnlineSpikingResNet', 'online_spiking_resnet18', 'online_spiking_resnet34', 'online_spiking_resnet50', 
           'online_spiking_resnext50_32x4d', 'online_spiking_wide_resnet50_2']

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
            if isinstance(module, (OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, SynapseNeuron, BasicBlock, Bottleneck)):
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


def _merge_synapse_neuron(synapse: Type[Union[nn.Conv2d, nn.Linear]] = None, neuron: callable = None, bn_dim: int = None, **kwargs):
    BN = config.args.BN
    if not config.args.BPTT:
        convNeuron = SynapseNeuron(synapse, neuron_class=neuron, v_reset=None, **kwargs)
        if BN:
            layers = [convNeuron]
        else:
            layers = [convNeuron, Scale(2.74)]
    else:
        if BN:
            bn = nn.BatchNorm2d(bn_dim)
            layers = [synapse, bn, neuron(decay_input = False, v_reset = None, tau=kwargs.get('tau', 2.0))]
        else:
            layers = [synapse, neuron(decay_input = False, v_reset = None, tau=kwargs.get('tau', 2.0)), Scale(2.74)]
    return SequentialModule(layers) if len(layers) > 1 else layers[0]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron: callable = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1 = conv3x3(inplanes, planes, stride)
        self.convNeuron1 = _merge_synapse_neuron(conv1, neuron, planes, **kwargs)
        conv2 = conv3x3(planes, planes)
        self.convNeuron2 = _merge_synapse_neuron(conv2, neuron, planes, **kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        identity = x

        out = self.convNeuron1(x, **kwargs)
        out = self.convNeuron2(out, **kwargs)
        
        if self.downsample is not None:
            identity = self.downsample(x, **kwargs)

        out = out + identity

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron: callable = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        conv1 = conv1x1(inplanes, width)
        self.convNeuron1 = _merge_synapse_neuron(conv1, neuron, width, **kwargs)
        conv2 = conv3x3(width, width, stride, groups, dilation)
        self.convNeuron2 = _merge_synapse_neuron(conv2, neuron, width, **kwargs)
        conv3 = conv1x1(width, planes * self.expansion)
        self.convNeuron3 = _merge_synapse_neuron(conv3, neuron, planes * self.expansion, **kwargs)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        identity = x

        out = self.convNeuron1(x, **kwargs)
        out = self.convNeuron2(out, **kwargs)
        out = self.convNeuron3(out, **kwargs)

        if self.downsample is not None:
            identity = self.downsample(x, **kwargs)

        out = out + identity

        return out


class OnlineSpikingResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron: callable = None,
        **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.convNeuron1 = _merge_synapse_neuron(conv1, neuron, self.inplanes, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], neuron=neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], neuron=neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], neuron=neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        neuron: callable = None,
        **kwargs
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_down = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample = _merge_synapse_neuron(conv_down, neuron, planes * block.expansion, **kwargs)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, neuron, **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    neuron=neuron,
                    **kwargs
                )
            )

        return SequentialModule(*layers)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.convNeuron1(x, **kwargs)
        x = self.maxpool(x)

        x = self.layer1(x, **kwargs)
        x = self.layer2(x, **kwargs)
        x = self.layer3(x, **kwargs)
        x = self.layer4(x, **kwargs)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_spike(self):
        raise NotImplementedError('get_spike not implemented now!')


def _online_spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
    model = OnlineSpikingResNet(block, layers, neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def online_spiking_resnet18(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_resnet34(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_resnet50(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_resnext50_32x4d(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _online_spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_wide_resnet50_2(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _online_spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)
