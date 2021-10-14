# 2021-10-5
# To decrease complexity and decrease numorous of params
# so taht the U-net could work on M300
# based on the strucutre https://ieeexplore.ieee.org/abstract/document/9319207
# attention not applied yet

from typing_extensions import Concatenate
import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.channelshuffle import ChannelShuffle
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# first conv, kernel_size = 7x7
# checked
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias = False), # kernel = 7, stride = 1, padding = 3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv0(x)

# depthwise Conv: use "SqueezeNet" to decrease the num of parameters
# downsqueeze1, replaced the convs, channels (55, 55) (27, 27), (13, 13)

# checked
class DWconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWconv, self).__init__()
        self.dwconv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1),
        )

    def forward(self, x):
        return self.dwconv(x)

class Fire(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Fire, self).__init__()
        self.downSQ_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias = False),
        )
        self.downSQ_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace = True),
            DWconv(out_channels, out_channels),
            nn.ChannelShuffle(2),
        )
    def forward(self, x):
        x = torch.cat((self.downSQ_1(x), self.downSQ_2(x)), 1)
        x = nn.ReLU()(x)
        return x
        
class Conv33(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv33, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # kernel = 3, stride = 1, padding = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.conv3(x)

class Conv11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv11, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False), # kernel = 1, stride = 1, padding = 0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.conv1(x)

class Squeezeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.stream1conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.stream2conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward_1(self, x):
        x_1 = self.inconv(x)
        x_1 = self.stream1conv(x_1)
        return x_1
    def forward_2(self, x):
        x_2 = self.inconv(x)
        x_2 = self.stream2conv(x_2)
        return x_2

    def forward(self, x):
        return torch.cat((self.forward_1(x), self.forward_2(x)), 1)

class deFire(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(deFire, self).__init__()
        self.upSQ = nn.Sequential(
            Squeezeconv(in_channels, out_channels),
            nn.ReLU(inplace= True),
            # re-sampling?
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )
    def forward(self, x):
        return self.upSQ(x)

# class upblock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(upblock, self).__init__()
#         self.up = nn.Sequential(
#             deFire(in_channels, out_channels*2),
#             Conv33(out_channels*2, out_channels)
#         )
#     def forward(self, x):
#         return self.up(x)

class finalconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(finalconv, self).__init__()
        self.final = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        return self.final(x)

class agskip(nn.Module): # attetnion gate skip connection
    def __init__(self, g_in, up_in):
        super(agskip, self).__init__()
        self.gateconv = nn.Conv2d(g_in, g_in, kernel_size=1, stride=1, padding=0)
        self.inputconv = nn.Conv2d(up_in, g_in, kernel_size=1, stride=1, padding=0)
        # add
        #relu
        self.psi = nn.Conv2d(g_in, g_in, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()
        # resample to x.size?
        # matmul x, out
    def forward(self, g_in, up_in):
        g_out = self.gateconv(g_in)
        up_out = self.inputconv(up_in)
        a_out = torch.add(g_out, up_out)
        x = nn.ReLU()(a_out)
        x = self.psi(x)
        x = self.sig(x)
        y = torch.matmul(x, up_in)
        return y

# added squeeze to replace the convs in the middle stages.
class unetlight(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 begin_channels = 64,
                 out_channels = 1,):
        super(unetlight, self).__init__()
        

        self.pools = nn.MaxPool2d(kernel_size=3, stride=1)

        # down sampling part
        self.down00 = Conv0(in_channels, begin_channels)
        # add pool

        self.down11 = Fire(begin_channels, 64)
        self.down12 = Fire(128, 64)
        # add skip
        self.down13 = Fire(128, 64)
        # add pools

        self.down21 = Fire(128, 128)
        # add skip
        self.down22 = Fire(256, 128)
        self.down23 = Fire(256, 128)
        self.down24 = Fire(256, 256)
        # add pools

        # final down sampling (bottle neck)
        self.down31 = Fire(512, 256)
        self.down32 = Conv11(512, 512)
        self.down33 = Conv33(512, 512)

        # skip connections

        # up sampling part
        self.ups11 = deFire(512, 128)
        self.ups12 = agskip(256, 512)
        self.ups13 = Conv33(512, 256)

        self.ups21 = deFire(256, 64)
        self.ups22 = agskip(128, 256)
        self.ups23 = Conv33(256, 128)

        self.ups31 = deFire(128, 32)
        self.ups32 = agskip(64, 128)
        self.ups33 = Conv33(128, 64)
        
        # final layer
        self.finalconv = finalconv(64, 1)

    def forward(self, x):
        x_00 = x
        x_01 = self.down00(x_00)
        x_skip1 = x_01
        x_02 = self.pools(x_01)
        print("down0 shape:", x_02.shape)

        x_10 = x_02
        x_11 = self.down11(x_10)
        x_12 = self.down12(x_11)
        x_skip2 = x_12
        x_13 = self.down13(x_12)
        x_14 = self.pools(x_13)
        print("down1 shape:", x_14.shape)

        x_20 = x_14
        x_21 = self.down21(x_20)
        x_skip3 = x_21
        x_22 = self.down22(x_21)
        x_23 = self.down23(x_22)
        x_24 = self.down24(x_23)
        x_25 = self.pools(x_24)
        print("down2 shape:", x_25.shape)

        x_30 = x_25
        x_31 = self.down31(x_30)
        x_32 = self.down32(x_31)
        x_33 = self.down33(x_32)
        print("down3 shape:", x_33.shape)

        print("skip1 shape:", x_skip1.shape)
        print("skip2 shape:", x_skip2.shape)
        print("skip3 shape:", x_skip3.shape)

        x_up3 = x_33
        x_up10 = x_33
        x_up111 = self.ups11(x_up10)
        x_up112 = self.ups12(x_skip3, x_up3)
        x_up12 = torch.cat((x_up111, x_up112), 1)
        x_up13 = self.ups13(x_up12)

        x_up20 = x_up13
        x_up2 = x_up13
        x_up211 = self.ups21(x_up20)
        x_up212 = self.ups22(x_skip2, x_up2)
        x_up22 = torch.cat((x_up211, x_up212), 1)
        x_up23 = self.ups23(x_up22)

        x_up30 = x_up23
        x_up1 = x_up23
        x_up311 = self.ups31(x_up30)
        x_up312 = self.ups(x_skip1, x_up1)
        x_up32 = torch.cat((x_up311, x_up312), 1)
        x_up33 = self.ups33(x_up32)

        x_out = self.finalconv(x_up33)
        y = x_out
        return y

# test whether net model works fine

def test():
    x = torch.randn(1, 3, 255, 255)
    print("input size", x.shape)
    model = unetlight()
    y = model(x)
    print(y.shape)

test()

# preds = model(img_tensor)

