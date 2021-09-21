#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Lee Ltd. All rights reserved.
#
#   @Filename: detection.py
#
#   @Author: Qiao Linhan
#
#   @Date: 2021-09-20
#
#   @Email: 742954173@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------

from typing_extensions import Concatenate
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # kernel = 3, stride = 1, padding = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, biase = False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel = 2, stride = 2) # list
        # down part of U-net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, out_channels))
            in_channels = feature

        # up part of U-net, use the transpose convolutions
        for feature in reversed(features): 
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel = 2, stride = 2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # bottle neck
        self.bottleneck = DoubleConv(features[-1], feature[-1]*2)
        # final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2): # step of 2, up double Conv, up, double conv...
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # what if input 161*161? there would be 80*80, then to 160*160
            if x.shape != skip_connection.shape:
                # add padding, resizing, ...
                x = TF.resize(x, size = skip_connection.shape[2:]) 


            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x) 

# test
def test():
    x = torch.randn((3, 1, 160, 160)) # batchsize = 3, channels = 1, inputsize = 160*160
    model = UNET(in_channels=1, out_channels= 1)
    preds = model(x)
    print('preds shape:', preds.shape)
    print('input shape:', x.shape)
    assert preds.shape == x.shape

if __name__ == "main":
    test() 