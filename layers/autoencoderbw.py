try:
    from .encoder import Encoder
    from .decoder import Decoder
except:
    from encoder import Encoder
    from decoder import Decoder
    
import torch.nn as nn


class AutoEncoderBW(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.encoder = Encoder(n, 1)
        self.decoder = Decoder(n, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    import os
    try:
        from torchinfo import summary
    except:
        os.system("pip install torchinfo")
        from torchinfo import summary
    autoencoder = AutoEncoderBW(16)
    print(summary(autoencoder, input_size=(24, 1, 1024, 768)))