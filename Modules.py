import torch
from torch import nn


class HaimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(3, 128),   nn.LeakyReLU(),
            nn.Linear(128, 256), nn.LeakyReLU(),
            nn.Linear(256, 512), nn.LeakyReLU(),
            nn.Linear(512, 512), nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def init_weights(self):
        def initializer(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.constant_(m.weight, 1)
                m.bias.data.fill_(0.1)
        self.linear_relu.apply(initializer)

    def forward(self, x):
        return self.linear_relu(x)


class Discriminator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.main = nn.Sequential()

    def forward(self, x):
        return self.main(x)
