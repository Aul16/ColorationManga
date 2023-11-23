from torch import nn

class Identity(nn.Module):
    def __init__(self, channel, activation=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, padding='same')
        self.batch1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 5, 1, padding='same')
        self.batch2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, 5, 1, padding='same')
        self.batch3 = nn.BatchNorm2d(channel)
        self.relu = nn.LeakyReLU(0.2)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch3(x)
        if self.activation:
            x = self.relu(x)
        return x