# 2021-10-5
# To decrease complexity and decrease numorous of params
# so taht the U-net could work on M300
# based on the strucutre https://ieeexplore.ieee.org/abstract/document/9319207
# attention not applied yet
#%%
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
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 1, bias = False), # kernel = 7, stride = 1, padding = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv0(x)

# depthwise Conv: use "SqueezeNet" to decrease the num of parameters
# downsqueeze1, replaced the convs, channels (55, 55) (27, 27), (13, 13)

class DWconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, stride=1, padding=1, groups=1)
        self.point_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, stride=1, padding=0, groups=1)

    def forward(x):
        x = DWconv(x)

class squeeze(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(squeeze, self).__init__()
        self.downSQ = nn.Sequential(
            # fire?
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False),
            nn.ReLU(inplace = True),
            # pip install torch-dwconv did not work well, could try again
            # dwconv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ChannelShuffle(5),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        out = self.downSQ(x)

class desqueeze(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(desqueeze, self).__init__()
        self.upSQ = nn.Sequential(
            # defire?
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 1, bias = False),
            nn.ReLU(inplace= True),
            # re-sampling?
            # ...
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )
    def forward(self, x):
        return self.upSQ(x)

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # kernel = 3, stride = 1, padding = 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv1(x)

# added squeeze to replace the convs in the middle stages.
class unetlight(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 begin_channels = 112,
                 out_channels = 1,
                 features = [55, 27, 13]):
        super(unetlight, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pools = nn.MaxPool2d(kernel_size=3, stride=3)

        # begin layer, conv0
        self.begin = Conv0(in_channels, begin_channels)

        # down sampling part
        # self.downs.append(Conv0(in_channels, begin_channels))
        for feature in features:
            self.downs.append(squeeze(begin_channels, feature))
            begin_channels = feature

        # up sampling part
        for feature in reversed(features):
            self.ups.append(desqueeze(feature*2, feature))
            self.ups.append(Conv1(feature*2, feature))

        # bottle neck
        self.bottleneck = Conv1(features[-1], features[-1]*2)

        # final layer
        self.final_conv_1 = Conv1(features[0], begin_channels)
        self.final_conv = nn.Conv2d(begin_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.begin(x)
        skip_connections.append(x)
        x = self.pools(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pools(x)

        x = self.bottleneck(x)


        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connections = skip_connections[idx//2]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size = skip_connections.shape[2:])


            concat_skip = torch.cat((skip_connections, x), dim = 1)
            x = self.ips[idx+1](concat_skip)

        return self.final_conv(x)

# img_path = "./wildfireeg001.jpg"
img = Image.open("wildfireeg001.jpg")
#%%
img_tensor = transforms.ToTensor()(img).unsqueeze(0)
plt.imshow(transforms.ToPILImage()(img_tensor))
model = unetlight()
#%%
preds = model(img_tensor)

