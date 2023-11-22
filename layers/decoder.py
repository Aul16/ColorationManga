from torch import nn
from .up_scale import up_scale

class Decoder(nn.Module):
    def __init__(self, n, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up3 = up_scale(n, n)
        self.up4 = up_scale(n, n)
        self.up5 = up_scale(n, 1, False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x3 = self.up3(x)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x5 = self.tanh(x5)
        return x5