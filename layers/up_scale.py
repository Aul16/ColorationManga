from torch import nn
from .identity import Identity


class up_scale(nn.Module):
    def __init__(self, channel_in, channel_out, activation=True):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(channel_in, channel_out, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(channel_in, channel_out, 5, stride=2, padding=2, output_padding=1)
        self.batch_up1 = nn.BatchNorm2d(channel_out)
        self.batch_up2 = nn.BatchNorm2d(channel_out)
        self.id1 = nn.Conv2d(channel_in, channel_in, 3, stride=1, padding='same')
        self.batch_id1 = nn.BatchNorm2d(channel_in)
        self.id2 = nn.Conv2d(channel_out, channel_out, 5, stride=1, padding='same')
        self.batch_id2 = nn.BatchNorm2d(channel_out)
        self.identity = Identity(channel_out, activation)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = x
        x = self.id1(x)
        x = self.batch_id1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.batch_up1(x)
        x = self.relu(x)
        x = self.id2(x)
        x = self.batch_id2(x)
        x = self.relu(x)
        x1 = self.up2(x1)
        x1 = self.batch_up2(x1)
        x1 = self.relu(x1)
        x = self.identity(x + x1)
        return x