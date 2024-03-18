import torch.nn as nn
import torch
from .down_scale import down_scale


class Discriminator(nn.Module):
    def __init__(self, channel_in, n):
        super().__init__()
        self.cnn = nn.Sequential(
            down_scale(channel_in, n),
            down_scale(n, n*2),
            down_scale(n*2, n*4),
            down_scale(n*4, n*8),
            down_scale(n*8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y=None):
        if y != None:
            x = torch.cat((x, y), dim=1)
        x = self.cnn(x)
        x = self.sigmoid(x)
        return x
    

if __name__ == "__main__":
    from torchinfo import summary
    model = Discriminator(4, 64)
    print(summary(model, ((1, 3, 512, 512), (1, 1, 512, 512))))