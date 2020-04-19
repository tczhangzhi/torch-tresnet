import torch
import torch.nn as nn

from functools import partial
from inplace_abn import InPlaceABN
from collections import OrderedDict

from .utils import load_state_dict_from_url
from .layers.squeeze_and_excite import SEModule
from .layers.avg_pool import FastGlobalAvgPool2d
from .layers.space_to_depth import SpaceToDepthModule
from .layers.anti_aliasing import AntiAliasDownsampleLayer

model_urls = {
    'tresnet_m': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_m.pth',
    'tresnet_m_448': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_m_448.pth',
    'tresnet_l': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_l.pth',
    'tresnet_l_448': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_l_448.pth',
    'tresnet_xl': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_xl.pth',
    'tresnet_xl_448': 'https://github.com/tczhangzhi/torch-tresnet/releases/download/v1.0.0/tresnet_xl_448.pth',
}


def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children():
        IABN2Float(child)
    return module


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes,
                                        planes,
                                        kernel_size=3,
                                        stride=2,
                                        activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu", activation_param=1e-3),
                    anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1, activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, remove_aa_jit=False):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock,
                                  self.planes,
                                  layers[0],
                                  stride=1,
                                  use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock,
                                  self.planes * 2,
                                  layers[1],
                                  stride=2,
                                  use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck,
                                  self.planes * 4,
                                  layers[2],
                                  stride=2,
                                  use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck,
                                  self.planes * 8,
                                  layers[3],
                                  stride=2,
                                  use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(
            OrderedDict([('SpaceToDepth', space_to_depth), ('conv1', conv1), ('layer1', layer1), ('layer2', layer2),
                         ('layer3', layer3), ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [
                conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, activation="identity")
            ]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


def TResnetM(num_classes, in_chans=3, remove_aa_jit=False):
    """ Constructs a medium TResnet model.
    """
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans, remove_aa_jit=remove_aa_jit)
    return model


def TResnetL(num_classes, in_chans=3, remove_aa_jit=False):
    """ Constructs a large TResnet model.
    """
    model = TResNet(layers=[4, 5, 18, 3],
                    num_classes=num_classes,
                    in_chans=in_chans,
                    width_factor=1.2,
                    remove_aa_jit=remove_aa_jit)
    return model


def TResnetXL(num_classes, in_chans=3, remove_aa_jit=False):
    """ Constructs an extra-large TResnet model.
    """
    model = TResNet(layers=[4, 5, 24, 3],
                    num_classes=num_classes,
                    in_chans=in_chans,
                    width_factor=1.3,
                    remove_aa_jit=remove_aa_jit)

    return model


def _tresnet(arch, layers, pretrained, progress, **kwargs):
    model = TResNet(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)['model']
        model.load_state_dict(state_dict)
    return model


def tresnet_m(pretrained=False, progress=True, **kwargs):
    """ Constructs a medium TResnet model.
    """
    return _tresnet('tresnet_m', [3, 4, 11, 3], pretrained, progress, **kwargs)


def tresnet_l(pretrained=False, progress=True, **kwargs):
    """ Constructs a large TResnet model.
    """
    return _tresnet('tresnet_l', [4, 5, 18, 3], pretrained, progress, width_factor=1.2, **kwargs)


def tresnet_xl(pretrained=False, progress=True, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    return _tresnet('tresnet_xl', [4, 5, 24, 3], pretrained, progress, width_factor=1.3, **kwargs)


def tresnet_m_448(pretrained=False, progress=True, **kwargs):
    """ Constructs a medium TResnet model.
    """
    return _tresnet('tresnet_m_448', [3, 4, 11, 3], pretrained, progress, **kwargs)


def tresnet_l_448(pretrained=False, progress=True, **kwargs):
    """ Constructs a large TResnet model.
    """
    return _tresnet('tresnet_m_448', [4, 5, 18, 3], pretrained, progress, width_factor=1.2, **kwargs)


def tresnet_xl_448(pretrained=False, progress=True, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    return _tresnet('tresnet_m_448', [4, 5, 24, 3], pretrained, progress, width_factor=1.3, **kwargs)