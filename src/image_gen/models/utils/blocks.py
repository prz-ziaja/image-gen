import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class activatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        num_groups=1,
        act_function=nn.GELU,
    ):
        nn.Module.__init__(self)

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
            ),
            nn.GroupNorm(num_groups, out_channels),
            act_function(),
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

        self.rearrange_expr = "b c (h rh) (w rw) -> b (c rh rw) h w"

        self.rearrange = Rearrange(self.rearrange_expr, rh=rh, rw=rw)
        self.conv = activatedConv2d(
            in_channels=rh * rw * input_channels,
            out_channels=input_channels,
            num_groups=group_size,
        )

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv(x)
        return x


class upBlock2d(nn.Module):
    def __init__(self, in_chs, out_chs, group_size, kernel_size=3, stride=1, padding=1):
        """
        Block consist of two parts:
         - ConvTranspose2d which upscales the input by 2 times
         - activatedConv2d which processes the output of ConvTranspose2d
        There is possibility to manipulate size of the output of the block by
        Overriding kernel_size, stride and padding of activatedConv2d.
        """
        nn.Module.__init__(self)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_chs, out_chs, 2, 2),
            activatedConv2d(
                in_channels=out_chs,
                out_channels=out_chs,
                num_groups=group_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activatedConv2d(
                in_channels=out_chs,
                out_channels=out_chs,
                num_groups=group_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class upBlock2dDoubleInput(upBlock2d):
    def __init__(
        self,
        in_chs_0,
        in_chs_1,
        out_chs,
        group_size,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        upBlock2d.__init__(
            self,
            in_chs=in_chs_0 + in_chs_1,
            out_chs=out_chs,
            group_size=group_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return super(upBlock2dDoubleInput, self).forward(x)


class downBlock2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, group_size=1
    ):
        """
        By default reduces height and width of the image by 2 times by utilizing rearrangePolling2d.
        """
        nn.Module.__init__(self)

        self.model = nn.Sequential(
            activatedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_groups=group_size,
            ),
            activatedConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_groups=group_size,
            ),
            rearrangePolling2d(out_channels, group_size),
        )

    def forward(self, x):
        return self.model(x)
