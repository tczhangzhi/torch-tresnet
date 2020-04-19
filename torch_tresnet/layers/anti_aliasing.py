import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels


        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)

        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt.to(input.device), stride=self.stride, padding=0, groups=input.shape[1])
