from torch import nn
try:
    from .up_scale import up_scale
    from .identity import Identity
except:
    from up_scale import up_scale
    from identity import Identity

class Decoder(nn.Module):
    def __init__(self, n, output_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up3 = up_scale(n, n)
        self.up4 = up_scale(n, n)
        self.up5 = up_scale(n, output_channel, False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x3 = self.up3(x)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x5 = self.tanh(x5)
        return x5
    

if __name__ == "__main__":
    import os
    try:
        from torchinfo import summary
    except:
        os.system("pip install torchinfo")
        from torchinfo import summary
    decoder = Decoder(48, 3)
    print(summary(decoder, input_size=(16, 8, 128, 96)))