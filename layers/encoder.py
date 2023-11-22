import torch.nn as nn
from .down_scale import down_scale

class Encoder(nn.Module):
    def __init__(self, n, input_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down1 = down_scale(input_channel, n)
        self.down2 = down_scale(n, n)
        self.down3 = down_scale(n, n, False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = self.tanh(x3)
        return x3