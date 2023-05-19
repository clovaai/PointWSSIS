"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .modules import ConvBNReLU, ASPP


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + mid_ch, mid_ch, kernel_size=1),
            ConvBNReLU(mid_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResNetUNet(nn.Module):
    def __init__(self, output_channel, input_channel):
        super().__init__()

        base_model = models.resnet101(pretrained=True)
        base_layers = list(base_model.children())

        if input_channel == 3:
            self.layer0 = nn.Sequential(*base_layers[0:3])
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                base_layers[1],
                base_layers[2],
            )
        
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer2 = base_layers[5]
        self.layer3 = base_layers[6]
        self.layer4 = base_layers[7]

        self.aspp = ASPP(2048, 512, atrous_rates=[3, 6, 9])

        self.decoder_3 = double_conv(1024, 512, 256)
        self.decoder_2 = double_conv(512, 256, 128)
        self.decoder_1 = double_conv(256, 128, 64)
        self.decoder_0 = double_conv(64, 64, 64)

        self.head = nn.Conv2d(64, output_channel, kernel_size=1, padding=0, bias=True)

        self.head_init()


    def head_init(self,):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)


    def forward(self, x):
        ori_size = x.size()[2:]

        enc_0 = self.layer0(x)     # [B, 64, H/2, W/2]
        enc_1 = self.layer1(enc_0) # [B, 256, H/4, W/4]
        enc_2 = self.layer2(enc_1) # [B, 512, H/8, W/8]
        enc_3 = self.layer3(enc_2) # [B, 1024, H/16, W/16]
        enc_4 = self.layer4(enc_3) # [B, 2048, H/32, W/32]

        x = self.aspp(enc_4)
        x = F.interpolate(x, size=enc_3.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_3], dim=1)
        x = self.decoder_3(x)

        x = F.interpolate(x, size=enc_2.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_2], dim=1)
        x = self.decoder_2(x)

        x = F.interpolate(x, size=enc_1.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_1], dim=1)
        x = self.decoder_1(x)

        x = F.interpolate(x, size=enc_0.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_0], dim=1)
        x = self.decoder_0(x)

        x = self.head(x)
        x = F.interpolate(x, size=ori_size, mode="bilinear", align_corners=False)

        return x


def UNet_ResNet101(
    output_channel=2,
    input_channel=3,
):
    model = ResNetUNet(
        output_channel=output_channel,
        input_channel=input_channel,
    )

    return model


if __name__ == "__main__":

    input_channel = 85
    output_channel = 2

    net = UNet(output_channel, input_channel).cuda()

    in_ten = torch.randn(1, input_channel, 256, 256).cuda()

    net.eval()
    out_ten = net(in_ten)

    print(out_ten.shape)
