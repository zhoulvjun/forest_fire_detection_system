import torch
import torch.nn as nn
from torch.nn.modules import batchnorm
from torch.nn.modules.activation import ReLU
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
from torchvision import transforms

# input conv 3-64
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride = 2, padding =3, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = Conv0(x)

# 1st sequential conv (64, 3) (128, 3) (256, 5) (512, 2)
class doulbelConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doulbelConv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = doulbelConv1(x)

# 2nd sequential conv (64, 128, 256, 512)
class doulbelConv2_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doulbelConv2_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 2, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = doulbelConv2_1(x)

class doulbelConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doulbelConv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = doulbelConv2(x)
        # out = doulbelConv2_1(out)

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding =1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = bottleneck(x)

# building the unet
class unetfastai(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, 
                 features = [64, 128, 256, 512]):
        super(unetfastai, self).__init__()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        
        
        self.downsampling.append(Conv0(in_channels, features[0]))
        for i in range(2): # 0 1 2
            self.downsampling.append(doulbelConv1(features[0], features[0]))
            i += 1
        self.downsampling.append(doulbelConv2(features[0], features[1]))
        for i in range(2): # 0 1 2
            self.downsampling.append(doulbelConv1(features[1], features[1]))
            i += 1
        self.downsampling.append(doulbelConv2(features[1], features[2]))
        for i in range(4): # 0 1 2 3 4
            self.downsampling.append(doulbelConv1(features[2], features[2]))
            i += 1
        self.downsampling.append(doulbelConv2(features[2], features[3]))


        # upsampling
        for feature in reversed(features):
            self.upsampling.append(nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2)
            )

        # bottle neck
        self.bottleneck = bottleneck(features[-1], features[-1]*2)
        # final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)

    def forward(self, x):
        # downsampling parts
        skip_connections = []

        x = self.downsampling(x)
        skip_connections.append(x)
        x = self.pools(x)

        for down in self.downsampling:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.upsampling), 2):
            x = self.upsampling[idx](x)
            skip_connections = skip_connections[idx//2]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size = skip_connections.shape[2:]) # -1 pixel


            concat_skip = torch.cat((skip_connections, x), dim = 1)
            x = self.ips[idx+1](concat_skip)

        return self.final_conv(x)

img_path = "/Users/qiao/dev/datas/wildfireeg001.jpg"
img = Image.open(img_path)
img_np = np.array(img)
img_tensor = transforms.ToTensor()(img).unsqueeze(0)
print(img_tensor.shape)

model = unetfastai()
preds = model(img_tensor)


