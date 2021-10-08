#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: LightUnetModel.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-08
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------
from FireModel import Fire
from DeFireModel import DeFire

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# first conv


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1, self).__init__()

        self.conv1 = nn.Sequential(
            # kernel = 7, stride = 1, padding = 3
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv1(x)


class unetlight(nn.Module):
    def __init__(self):

        super(unetlight, self).__init__()

        self.in_channels = 3
        self.begin_channels = 112
        self.out_channels = 1
        self.features = [55, 27, 13]

        # net
        self.conv1 = Conv1(self.in_channels, self.begin_channels)

        self.maxpool = nn.MaxPool2d(3, 1, 1)

        self.fire_f2 = Fire(self.begin_channels, self.features[0])
        self.fire_f3 = Fire(self.features[0], self.features[0])
        self.fire_f4 = Fire(self.features[0], self.features[0])

        self.fire_f5 = Fire(self.features[0], self.features[1])
        self.fire_f6 = Fire(self.features[1], self.features[1])
        self.fire_f7 = Fire(self.features[1], self.features[1])
        self.fire_f8 = Fire(self.features[1], self.features[1])

        self.fire_f9 = Fire(self.features[1], self.features[2])

        self.double_conv = nn.Sequential(
            nn.Conv2d(self.features[2], self.features[2], 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(self.features[2], self.features[2], 3, 1, 1),
            nn.ReLU(),
        )

        self.defire_d1 = DeFire(self.features[2], self.features[1])
        self.half_conv1 = nn.Sequential(
                nn.Conv2d( self.features[1]*2, self.features[1], 3, 1, 1),
                nn.ReLU()
                )
        self.defire_d2 = DeFire(self.features[1], self.features[0])
        self.half_conv2 = nn.Sequential(
                nn.Conv2d( self.features[0]*2, self.features[0], 3, 1, 1),
                nn.ReLU()
                )
        self.defire_d3 = DeFire(self.features[0], self.begin_channels)
        self.half_conv3 = nn.Sequential(
                nn.Conv2d( self.begin_channels*2, self.begin_channels, 3, 1, 1),
                nn.ReLU()
                )

        self.out_conv = nn.Sequential(
                nn.Conv2d( self.begin_channels, self.out_channels, 3, 1, 1),
                nn.ReLU()
                )


    def forward(self, x):

        conv1_out = self.conv1(x)  # TODO: need to jump

        max_p1_out = self.maxpool(conv1_out)

        f2_out = self.fire_f2(max_p1_out)
        f3_out = self.fire_f3(f2_out)
        f4_out = self.fire_f4(f3_out)

        max_p2_out = self.maxpool(f4_out)

        f5_out = self.fire_f5(max_p2_out)
        f6_out = self.fire_f5(f5_out)
        f7_out = self.fire_f5(f6_out)
        f8_out = self.fire_f5(f7_out)

        max_p3_out = self.maxpool(f8_out)

        f9_out = self.fire_f9(max_p3_out)

        double_conv_out = self.double_conv(f9_out)

        d1_out = self.defire_d1(double_conv_out)

        half_conv_1_out = self.half_conv1(torch.cat([double_conv_out, d1_out],1))

        d2_out = self.defire_d2(half_conv_1_out)

        half_conv_2_out = self.half_conv2(torch.cat([f3_out, d2_out],1))

        d3_out = self.defire_d3(half_conv_2_out)

        half_conv_3_out = self.half_conv3(torch.cat([conv1_out, d3_out], 1))

        return self.out_conv(half_conv_3_out)
