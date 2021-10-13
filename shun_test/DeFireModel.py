import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class DeFire(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeFire, self).__init__()
        self.upSQ = nn.Sequential(
            # defire?
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.ReLU(inplace= True),
            # re-sampling?
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )
    def forward(self, x):
        return self.upSQ(x)

# # add squeeze and modified __qiao
# class Squeezeconv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.stream1conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.stream2conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward_1(self, x):
#         x_1 = self.inconv(x)
#         x_1 = self.stream1conv(x_1)
#         return x_1
#     def forward_2(self, x):
#         x_2 = self.inconv(x)
#         x_2 = self.stream2conv(x_2)
#         return x_2

#     def forward(self, x):
#         return torch.cat((self.forward_1(x), self.forward_2(x)), 1)

# class deFire(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(deFire, self).__init__()
#         self.upSQ = nn.Sequential(
#             Squeezeconv(in_channels, out_channels),
#             nn.ReLU(inplace= True),
#             # re-sampling?
#             nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
#         )
#     def forward(self, x):
#         return self.upSQ(x)