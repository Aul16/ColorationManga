import torch.nn as nn
import torch
try:
    from .down_scale import down_scale
except:
    from down_scale import down_scale

class Discriminator(nn.Module):
    def __init__(self, channel_in, n):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            down_scale(channel_in, n),
            down_scale(n, n*2),
            down_scale(n*2, n*4),
            down_scale(n*4, n*8),
            down_scale(n*8, 1)
        )
        self.linear = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    

if __name__ == "__main__":
    from torchinfo import summary
    model = Discriminator(32, 16)
    print(summary(model, ((8, 24, 128, 96), (8, 8, 128, 96))))