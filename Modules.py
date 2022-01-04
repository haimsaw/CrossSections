import math
import torch
from torch import nn
from itertools import chain


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def initializer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias, 0, 1)
        # m.bias.data.fill_(0.1)


def siren_initializer(m):
    if isinstance(m, nn.Linear):
        w_std = math.sqrt(6 / m.in_features)
        torch.nn.init.uniform_(m.weight, -w_std, w_std)
        torch.nn.init.uniform_(m.bias, -w_std, w_std)


class HaimNet(nn.Module):
    def __init__(self, hidden_layers, residual_module, embedder, is_siren):
        super().__init__()

        self.embedder = embedder
        self.is_siren = is_siren

        assert residual_module is None or next(residual_module.parameters()).requires_grad is False
        self.residual_module = residual_module

        n_neurons = [self.embedder.out_dim] + hidden_layers + [1]

        neurons = [nn.Linear(n_neurons[i], n_neurons[i + 1]) for i in range(len(n_neurons) - 2)]
        activations = [Sine() if is_siren else nn.LeakyReLU() for _ in range(len(n_neurons) - 2)]
        layers = list(chain.from_iterable(zip(neurons, activations))) + [nn.Linear(n_neurons[-2], 1)]

        self.function = nn.Sequential(*layers)
        self.first_layer = neurons[0]

    def init_weights(self):
        if self.is_siren:
            self.function.apply(siren_initializer)

            w_std = 1 / self.first_layer.in_features
            torch.nn.init.uniform_(self.first_layer.weight, -w_std, w_std)
            torch.nn.init.uniform_(self.first_layer.bias, -w_std, w_std)

        else:
            self.function.apply(initializer)

    def forward(self, xyzs):
        embbeded = self.embedder(xyzs)
        if self.residual_module is not None:
            # todo haim - is this true? I use the sigmoid on the right but not on the left?
            return self.function(embbeded) + torch.sigmoid(self.residual_module(xyzs))
        else:
            return self.function(embbeded)


# https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L48
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def __call__(self, x):
        return self.embed(x)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj
