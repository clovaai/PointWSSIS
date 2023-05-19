"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import torch
import torch.nn as nn

class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        padding=None,
        bias=False,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        padding=None,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class ConvAct(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        bias=False,
        padding=None,
        activation=nn.ReLU,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=bias,
            ),
            activation(),
        )

class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        bias=False,
        padding=None,
        activation=nn.ReLU,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            activation(),
        )

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        rate1, rate2, rate3 = tuple(atrous_rates)
        
        modules = []
        modules.append(ASPPPooling(in_channels, out_channels))
        modules.append(ConvBNReLU(in_channels, out_channels, 1, padding=0, dilation=1))
        modules.append(ConvBNReLU(in_channels, out_channels, 3, padding=rate1, dilation=rate1))
        modules.append(ConvBNReLU(in_channels, out_channels, 3, padding=rate2, dilation=rate2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3, padding=rate3, dilation=rate3))
        self.convs = nn.ModuleList(modules)

        self.project = ConvBNReLU(out_channels*len(modules), out_channels, 1, padding=0)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvBNReLU(
            inplanes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = ConvBN(planes, planes, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvBNReLU(inplanes, width, kernel_size=1, padding=0)
        self.conv2 = ConvBNReLU(
            width,
            width,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=groups,
            dilation=dilation,
        )
        self.conv3 = ConvBN(width, planes * self.expansion, kernel_size=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

