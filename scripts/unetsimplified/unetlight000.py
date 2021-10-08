# 2021-10-5
# To decrease complexity and decrease numorous of params
# so taht the U-net could work on M300
# based on the strucutre https://ieeexplore.ieee.org/abstract/document/9319207
# attention not applied yet

from typing_extensions import Concatenate
import torch
import torch.nn as nn
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

<<<<<<< HEAD
class Fire(nn.Module):
=======
# checked
class squeeze(nn.Module):
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec
    def __init__(self, in_channels, out_channels):
        super(Fire, self).__init__()
        self.downSQ = nn.Sequential(
            # fire?
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace = True),
            DWconv(out_channels, out_channels),
            nn.ChannelShuffle(5),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.downSQ(x)

<<<<<<< HEAD
class Conv33(nn.Module):
=======
# checked
class Conv1(nn.Module):
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec
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

<<<<<<< HEAD
class deFire(nn.Module):
=======
# checked
class desqueeze(nn.Module):
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec
    def __init__(self, in_channels, out_channels):
        super(deFire, self).__init__()
        self.upSQ = nn.Sequential(
            # defire?
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.ReLU(inplace= True),
            # re-sampling?
            # ...
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
        self.final = nn.Sequential(
<<<<<<< HEAD
            Conv11(in_channels, out_channels),
=======
            Conv1(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec
        )

    def forward(self, x):
        return self.final(x)


# added squeeze to replace the convs in the middle stages.
class unetlight(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 begin_channels = 112,
                 out_channels = 1,
                 features = [55, 27, 13]):
        super(unetlight, self).__init__()
        self.ups = nn.ModuleList()
<<<<<<< HEAD
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
=======
        self.downs = nn.ModuleList()

        # TODO:
        self.pools = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # begin layer, conv0
        self.begin = Conv0(in_channels, begin_channels)

        # down sampling part
        # self.downs.append(Conv0(in_channels, begin_channels))
        # 1
        self.downs.append(squeeze(begin_channels, features[0]))
        for i in range(1):
            self.downs.append(squeeze(features[0],features[0]))
        self.downs.append(self.pools)
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec

        # up sampling part
        for feature in reversed(features):
            self.ups.append(upblock(feature*2, feature))
        self.ups.append(upblock(features[0], begin_channels))


        # # bottle neck
        # self.bottleneck = Conv1(features[-1], features[-1]*2)

        # final layer
        self.finalconv = finalconv(features[0], in_channels)
    def forward(self, x):
        skip_connections = []

        x = self.down00(x)
        skip_connections.append(x)
        x = self.pools(x)

        x = self.down01(x)
        skip_connections.append(x)

        x = self.down02(x)
        skip_connections.append(x)
        x = self.pools(x)

        x = self.down03(x)
        x = self.pools(x)

        x = self.down04(x)
        skip_connections.append(x)
        x = self.pools(x)

        x = self.down05(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 1):
            x = self.ups[idx](x)
            skip_connections = skip_connections[idx]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size = skip_connections.shape[2:])


            concat_skip = torch.cat((skip_connections, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
<<<<<<< HEAD
=======

        return self.final_conv(x)
>>>>>>> e1b54eabcdc4b4a3055ca328bc2e432585a2bcec

        return self.finalconv(x)

# test whether net model works fine
# load the image
img_path = "datas/wildfireeg001.jpg"
img = Image.open(img_path)
img_tensor = transforms.ToTensor()(img).unsqueeze(0)
print(img_tensor)
# img_tensor = transforms.ToTensor()(img)
plt.imshow(transforms.ToPILImage()(img_tensor))
model = unetlight()

preds = model(img_tensor)

