import torch
import torch.nn as nn
from torch.autograd import Function
from modules.layer import ScaledWSConv2d, ScaledWSLinear, SWSConvNeuron, SWSLinearNeuron
from modules.neuron import OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, MyLIFNode
from modules import neuron_spikingjelly
import config

__all__ = [
    'OnlineSpikingVGG', 'online_spiking_vgg11', 'online_spiking_vgg11_ws', 'online_spiking_vgg11f_ws',
]

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


class SequentialModule(nn.Sequential):

    def __init__(self, single_step_neuron, *args):
        super(SequentialModule, self).__init__(*args)
        self.single_step_neuron = single_step_neuron

    def forward(self, input, **kwargs):
        for module in self._modules.values():
            if isinstance(module, (OnlineIFNode, OnlineLIFNode, OnlinePLIFNode, SWSLinearNeuron, SWSConvNeuron)):
                input = module(input, **kwargs)
            else:
                if isinstance(module, neuron_spikingjelly.BaseNode):
                    input = module.single_step_forward(input)
                else:
                    input = module(input)
        return input

    # def get_spike(self):
    #     spikes = []
    #     for module in self._modules.values():
    #         if isinstance(module, self.single_step_neuron):
    #             spike = module.spike.cpu()
    #             spikes.append(spike.reshape(spike.shape[0], -1))
    #     return spikes


class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale


class OnlineSpikingVGG(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True,
                 single_step_neuron: callable = None, light_classifier=True, BN=False, **kwargs):
        print(f'WS is {weight_standardization}, BN is {BN}')
        super(OnlineSpikingVGG, self).__init__()
        self.single_step_neuron = single_step_neuron
        # self.grad_with_rate = kwargs.get('grad_with_rate', False)  # always make it false
        self.fc_hw = kwargs.get('fc_hw', 3)
        self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization,
                                         neuron=single_step_neuron, BN=BN, **kwargs)
        if light_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
            self.classifier = SequentialModule(
                single_step_neuron, # not in the module, but parameter
                nn.Linear(512*(self.fc_hw**2), num_classes),
            )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
            linear_dim = min(4096, self.self.fc_hw ** 2)
            self.classifier = SequentialModule(
                single_step_neuron,
                SWSLinearNeuron(512 * self.fc_hw ** 2, linear_dim, neuron_class=single_step_neuron, **kwargs),
                Scale(2.74),
                nn.Dropout(),
                SWSLinearNeuron(linear_dim, linear_dim, neuron_class=single_step_neuron, **kwargs),
                Scale(2.74),
                nn.Dropout(),
                nn.Linear(linear_dim, num_classes),
            )
        if init_weights:
            self._initialize_weights()
    
    def reset_v(self):
        for module in self.features._modules.values():
            if isinstance(module, neuron_spikingjelly.BaseNode):
                module.v = 0.
        for module in self.classifier._modules.values():
            if isinstance(module, neuron_spikingjelly.BaseNode):
                module.v = 0.

    def forward(self, x, **kwargs):
        x = self.features(x, **kwargs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x, **kwargs)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: callable = None, BN=False, **kwargs):
        layers = []
        in_channels = kwargs.get('c_in', 3)
        use_stride_2 = False
        for v in cfg:
            if v == 'M':
                #layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif v == 'S':
                use_stride_2 = True
            else:
                if use_stride_2:
                    stride = 2
                    use_stride_2 = False
                else:
                    stride = 1
                # neuron = OnlineLIFNode(tau = 2., surrogate_function = surrogate.Sigmoid(), dropout = 0.0)
                if not config.args.BPTT:
                    convNeuron = SWSConvNeuron(in_channels, v, kernel_size=3, padding=1, stride=stride, neuron_class=neuron, **kwargs)
                    if BN:
                        bn = nn.BatchNorm2d(v)
                        layers += [convNeuron, bn]
                    else:
                        layers += [convNeuron, Scale(2.74)]
                else:
                    conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                    if BN:
                        bn = nn.BatchNorm2d(v)
                        layers += [conv2d, bn, neuron(decay_input = False, v_reset = None)]
                    else:
                        layers += [conv2d, neuron(decay_input = False, v_reset = None), Scale(2.74)]
                in_channels = v
        return SequentialModule(neuron, *layers)

    def get_spike(self):
        spikes = []
        for module in self.features._modules.values():
            if isinstance(module, self.single_step_neuron):
                spike = module.spike.cpu()
                spikes.append(spike.reshape(spike.shape[0], -1))
            if isinstance(module, SWSConvNeuron) or isinstance(module, SWSLinearNeuron):
                spike = module.neuron.spike.cpu()
                spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes



cfgs = {
    'A': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],#, 'M'],
}


def _spiking_vgg(arch, cfg, weight_standardization, pretrained, progress, single_step_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = OnlineSpikingVGG(cfg=cfgs[cfg], weight_standardization=weight_standardization, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def online_spiking_vgg11(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _spiking_vgg('vgg11', 'A', False, pretrained, progress, single_step_neuron, **kwargs)


def online_spiking_vgg11_ws(pretrained=False, progress=True, single_step_neuron: callable = None, weight_standardization=True, **kwargs):
    return _spiking_vgg('vgg11', 'A', weight_standardization, pretrained, progress, single_step_neuron, **kwargs)

