from .down_scale import down_scale
from .up_scale import up_scale


import torch.nn as nn


class UResNet(nn.Module):
    def __init__(self, n, channel_in, channel_out):
        super().__init__()
        self.down1 = down_scale(channel_in, n)
        self.down2 = down_scale(n, n * 2)
        self.down3 = down_scale(n * 2, n * 4)
        self.down4 = down_scale(n * 4, n * 8)
        self.down5 = down_scale(n * 8, n * 16)
        self.up1 = up_scale(n * 16, n * 8)
        self.up2 = up_scale(n * 8, n * 4)
        self.up3 = up_scale(n * 4, n * 2)
        self.up4 = up_scale(n * 2, n)
        self.up5 = up_scale(n, channel_out, False)
        self.tanh = nn.Tanh()

    def forward(self, entry):
        x1 = self.down1(entry)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        entry = self.up1(x5)
        entry = self.up2(entry + x4)
        entry = self.up3(entry + x3)
        entry = self.up4(entry + x2)
        entry = self.up5(entry + x1)
        entry = self.tanh(entry)
        return entry
