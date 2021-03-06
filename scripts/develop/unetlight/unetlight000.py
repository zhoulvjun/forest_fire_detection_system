# 2021-10-5
# To decrease complexity and decrease numorous of params
# so that the U-net could work on M300
# based on the strucutre https://ieeexplore.ieee.org/abstract/document/9319207
# 2021-11-11: added squeeze structure and attention gate

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

class upSqueezeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = torch.cat((self.conv1(x), self.conv3(x)), 1)
        return x

class deFire(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deFire, self).__init__()
        self.upsqueeze = upSqueezeconv(in_channels, out_channels)
        self.conv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        x = self.upsqueeze(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class finalconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(finalconv, self).__init__()
        self.final = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        return self.final(x)

class agskip(nn.Module): # attetnion gate skip connection
    def __init__(self, up_in, up_out):
        super(agskip, self).__init__()
        self.upconv = nn.Conv2d(up_in, up_in/2, kernel_size=1, stride=1, padding=0)
        self.gateconv = nn.Conv2d(up_in/2, up_in/2, kernel_size=1, stride=1, padding=0)
        self.add = torch.add()
        #relu
        self.psi = nn.Conv2d(up_in/2, up_out, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()
        # resample to x.size?
        # matmul x, out
    def forward(self, x_g, x_up):
        x_up = self.upconv(x_up)
        x_g = self.gateconv(x_g)
        x_add = self.add()(x_up, x_g)
        x = nn.ReLU()(x_add)
        x = self.psi(x)
        x = self.sig(x)
        y = torch.matmul(x, x)
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
        self.ups11 = deFire(512, 256)
        # self.ups12 = agskip(512, 512)
        self.ups13 = Conv33(512, 256)

        self.ups21 = deFire(256, 128)
        # self.ups22 = agskip(256, 256)
        self.ups23 = Conv33(256, 128)

        self.ups31 = deFire(128, 64)
        # self.ups32 = agskip(128, 128)
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
        print("up3 shape:", x_up3.shape)

        x_up10 = x_33
        print("up in shape:", x_up10.shape)

        x_up111 = self.ups11(x_up10)
        x_up112 = agskip(x_skip3, x_up3)
        x_up12 = torch.cat((x_up111, x_up112), 1)
        x_up13 = self.ups13(x_up12)

        x_up20 = x_up13
        x_up2 = x_up13
        x_up211 = self.ups21(x_up20)
        x_up212 = agskip(x_skip2, x_up2)
        x_up22 = torch.cat((x_up211, x_up212), 1)
        x_up23 = self.ups23(x_up22)

        x_up30 = x_up23
        x_up1 = x_up23
        x_up311 = self.ups31(x_up30)
        x_up312 = agskip(x_skip1, x_up1)
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

