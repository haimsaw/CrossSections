import torch
from torch import nn
from itertools import chain
from Helpers import *


def initializer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.constant_(m.weight, 1)
        m.bias.data.fill_(0.1)
    elif isinstance(m, nn.LeakyReLU) or isinstance(m, nn.Sequential):
        return
    else:
        assert False


class HaimNet(nn.Module):
    def __init__(self, n_neurons, residual_module):
        super().__init__()
        assert n_neurons[0] == 3 and n_neurons[-1] == 1

        neurons = [nn.Linear(n_neurons[i], n_neurons[i + 1]) for i in range(len(n_neurons) - 2)]
        activations = [nn.LeakyReLU() for i in range(len(n_neurons) - 2)]
        layers = list(chain.from_iterable(zip(neurons, activations))) + [nn.Linear(n_neurons[-2], 1)]
        self.linear_relu = nn.Sequential(*layers)

        # todo assert residual_module is None or residual_module.parameters()[0].requires_grad == False
        self.residual_module = residual_module

    def init_weights(self):
        self.linear_relu.apply(initializer)

    def forward(self, xyz):
        if self.residual_module is not None:
            return self.linear_relu(xyz) + self.residual_module(xyz)
        else:
            return self.linear_relu(xyz)


class HaimnetOctnetree(nn.Module):
    def __init__(self, haimnet_modules, octanes):
        super().__init__()

        self.haimnet_modules = haimnet_modules
        self.octanes = octanes

    def forward(self, xyz):

        [is_in_octant_tensor(xyz, octant) for haimnet_module, octant in zip(self.haimnet_modules, self.octanes)]
        # todo use torch.where?


        for haimnet_module, octant in zip(self.haimnet_modules, self.octanes):
            if is_in_octant_tensor(xyz, octant):
                return haimnet_module(xyz)
        raise Exception(f"xyz not in any octant. xyz={xyz}")
