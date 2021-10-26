import torch
from torch import nn
from itertools import chain


def initializer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.constant_(m.weight, 1)
        m.bias.data.fill_(0.1)
    else:
        assert False


class HaimNet(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        neurons = [nn.Linear(n_neurons[i], n_neurons[i + 1]) for i in range(len(n_neurons) - 1)]
        activations = [nn.LeakyReLU() for i in range(len(n_neurons) - 1)]
        layers = list(chain.from_iterable(zip(neurons, activations))) + [nn.Linear(n_neurons[-1], 1)]
        self.linear_relu = nn.Sequential(*layers)

    def init_weights(self):
        self.linear_relu.apply(initializer)

    def forward(self, x):
        return self.linear_relu(x)
