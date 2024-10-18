import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class geluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, num_groups=1):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.GroupNorm(num_groups, self.out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)

class rearrangePolling2d(nn.Module):
    def __init__(self, input_channels, group_size, rh=2, rw=2):
        """
        height of input must be divisable by rh
        widht of input must be divisable by rw
        """
        nn.Module.__init__(self)

        self.rh = rh
        self.rw = rw
        self.input_channels = input_channels
        self.ein_expr = "b c (h rh) (w rw) -> b (c rh rw) h w"

        self.rearrange = Rearrange(self.ein_expr, rh=rh, rw=rw)
        self.conv = geluConv2d(in_channels=rh*rw*input_channels, out_channels=input_channels, num_groups=group_size)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv(x)
        return x

class upBlock2dWithSkip(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        nn.Module.__init__(self)

        layers = [
            nn.ConvTranspose2d(2 * in_chs, out_chs, 2, 2),
            geluConv2d(in_channels=out_chs, out_channels=out_chs, num_groups=group_size),
            geluConv2d(in_channels=out_chs, out_channels=out_chs, num_groups=group_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
    
class upBlock2d(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        nn.Module.__init__(self)

        layers = [
            nn.ConvTranspose2d(in_chs, out_chs, 2, 2),
            geluConv2d(in_channels=out_chs, out_channels=out_chs, num_groups=group_size),
            geluConv2d(in_channels=out_chs, out_channels=out_chs, num_groups=group_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class downBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, group_size=1):
        nn.Module.__init__(self)

        layers = [
            geluConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, num_groups=group_size),
            geluConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, num_groups=group_size),
            rearrangePolling2d(out_channels, group_size)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
