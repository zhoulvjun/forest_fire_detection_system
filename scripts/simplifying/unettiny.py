# 2021-10-4
# simplify the U-net for woring on M300

from typing_extensions import Concatenate
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF

# down sampling part for tiny U-net
class downconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downconv, self).__init__
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        return self.down(x)

class unettiny(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 out_channels = 1,
                 features = [64, 128, 256]):
        super(unettiny, self).__init__
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pools = nn.MaxPool2d(kernel_size=3, stride=3)

        # down sampling part
        for feature in features:
            self.downs.append(in_channels, feature)
            in_channels = feature

        # up sampling part
        for upouts in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2))
            self.ups.append(downconv(feature*2, feature))

        # bottle neck
        self.bottleneck = downconv(features[-1], features[-1]*2)

        # final layer
        self.final_conv = nn.Conv2d(feature[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

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

# test model
def test(x):
    x = torch.random((1, 3, 255, 255))
    model = unettiny(in_channels=3, out_channels=1)
    preds = model(x)
    print(x.shape, preds.shape)

if __name__ == "main":
    test()



