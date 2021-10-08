import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class DWconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWconv, self).__init__()
        self.dwconv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1),
        )

    def forward(self, x):
        return self.dwconv(x)

# NOTE: squeeze is 1x1 conv


class Fire(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fire, self).__init__()
        self.fire = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace = True),
                DWconv(out_channels, out_channels),
                nn.ReLU(inplace = True),
                )
    def forward(self, x):
        return self.fire(x)

