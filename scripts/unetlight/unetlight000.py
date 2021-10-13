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
            nn.ChannelShuffle(5),
        )
    def forward(self, x):
        x = torch.cat((self.downSQ_1(x), self.downSQ_2(x)), 1)
        x = nn.ReLU(x)
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
        return self.conv11(x)

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

class upblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upblock, self).__init__()
        self.up = nn.Sequential(
            deFire(in_channels, out_channels*2),
            Conv33(out_channels*2, out_channels)
        )
    def forward(self, x):
        return self.up(x)

class finalconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(finalconv, self).__init__()
        self.final = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        return self.final(x)

class agskip(nn.Module): # attetnion gate skip connection
    def __init__(self, gate_in, gate_out, up_in, up_out, skip_in, skip_out):
        super(agskip, self).__init__()
        self.gateconv = nn.Conv2d(gate_in, gate_out, kernel_size=1, stride=1, padding=0)
        self.inputconv = nn.Conv2d(up_in, up_out, kernel_size=1, stride=2, padding=0)
        # add
        #relu
        self.psi = nn.Conv2d(skip_in, skip_out, kernel_size=1, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        # resample to x.size?
        # matmul x, out
    def forward(self, x_g, x_up):
        x_gout = self.gateconv(x_g)
        x_upout = self.inputconv(x_up)
        skip_out = torch.add(x_gout, x_upout)
        skip_out = nn.ReLU(skip_out)
        skip_out = self.psi(skip_out)
        skip_out = self.sigmoid(skip_out)
        y = torch.matmul(skip_out, x_up)
        return y

# added squeeze to replace the convs in the middle stages.
class unetlight(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 begin_channels = 64,
                 out_channels = 1,
                 features = [64, 128, 256]):
        super(unetlight, self).__init__()
        self.ups = nn.ModuleList()

        self.pools = nn.MaxPool2d(kernel_size=3, stride=1)

        # down sampling part
        # add pools, skip
        self.down00 = Conv0(in_channels, begin_channels)
        self.down01 = nn.ModuleList()
        # self.down02 = nn.ModuleList()
        # self.down03 = nn.ModuleList()
        self.down04 = nn.ModuleList()
        self.down05 = nn.ModuleList()

        # add skip
        self.down01.append(Fire(begin_channels, features[0]))
        self.down01.append(Fire(features[0], features[0]))
        # add pools
        self.down02 = Fire(features[0], features[0])
        # add skip
        self.down03 = Fire(features[0], features[1])
        # add pools
        self.down04.append(Fire(features[1], features[1]))
        self.down04.append(Fire(features[1], features[1]))
        self.down04.append(Fire(features[1], features[1]))
        # final down sampling (bottle neck)
        self.down05.append(Fire(features[1], features[2]))
        self.down05.append(Conv11(features[2], features[2]))
        self.down05.append(Conv33(features[2], features[2]))

        # up sampling part
        for feature in reversed(features):
            self.ups.append(upblock(feature*2, feature))
            self.ups.append(upblock(features[0], begin_channels))

        # final layer
        self.finalconv = finalconv(begin_channels, out_channels)

    def forward(self, x):
        x_g = [] # gate signals
        # 255-112, 3-64
        x = self.down00(x)
        x_g.append(x)
        x = self.pools(x)
        # 112-55, 64-64
        x = self.down01(x)
        x_g.append(x)

        x = self.down02(x)
        x_g.append(x)
        x = self.pools(x)
        # 55-27, 64-128
        x = self.down03(x)
        x = self.pools(x)
        # 27-13, 128-256
        x = self.down04(x)
        x_g.append(x)
        x = self.pools(x)
        # 13-13
        x = self.down05(x)

        x_g = x_g[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # TODO
            x_out = self.ups[idx](x) # input_signals, there might be problems
            skip_connections = agskip(x_g, x_out)[idx]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size = skip_connections.shape[2:])

            concat_skip = torch.cat((skip_connections, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.finalconv(x)

# test whether net model works fine

def test():
    x = torch.randn(1, 3, 255, 255)
    print(x.shape)
    model = unetlight()
    y = model(x)
    print(y)

test()

# preds = model(img_tensor)

