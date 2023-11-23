import torch.nn as nn
try:
    from .down_scale import down_scale
    from .identity import Identity
except:
    from down_scale import down_scale
    from identity import Identity

class Encoder(nn.Module):
    def __init__(self, n, input_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down1 = down_scale(input_channel, n)
        self.identity = Identity(n)
        self.down2 = down_scale(n, n)
        self.identity1 = Identity(n)
        self.down3 = down_scale(n, n)
        self.identity2 = Identity(n, False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.identity(x1)
        x2 = self.down2(x1)
        x2 = self.identity1(x2)
        x3 = self.down3(x2)
        x3 = self.identity2(x3)
        x3 = self.tanh(x3)
        return x3
    
if __name__ == "__main__":
    import os
    try:
        from torchinfo import summary
    except:
        os.system("pip install torchinfo")
        from torchinfo import summary
    encoder = Encoder(8, 3)
    print(summary(encoder, input_size=(8, 3, 1024, 768)))