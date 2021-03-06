import math
import torch
from torch import nn
from itertools import chain
from math import pi as PI
from hp import args


def initializer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias, 0, 1)
        # m.bias.data.fill_(0.1)


def get_MLP_layers(n_neurons):
    neurons = [nn.Linear(n_neurons[i], n_neurons[i + 1]) for i in range(len(n_neurons) - 2)]
    activations = [nn.LeakyReLU() for _ in range(len(n_neurons) - 2)]
    layers = list(chain.from_iterable(zip(neurons, activations))) + [nn.Linear(n_neurons[-2], n_neurons[-1])]
    return layers


class HaimNetWithState(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.xyz_embedder = get_embedder(args.num_embedding_freqs, hp.spherical_coordinates, disable_learnable=args.disable_learnable_embed)
        n_freqs = int(args.hidden_state_size/2)
        self.hidden_state_embedder = get_embedder(n_freqs, input_dims=1, include_input=False)\
            if hp.hidden_state_embedder else None

        assert self.hidden_state_embedder is None or self.hidden_state_embedder.out_dim == args.hidden_state_size
        n_neurons = [self.xyz_embedder.out_dim + 1 + args.hidden_state_size] + hp.hidden_layers + [1 + args.hidden_state_size]

        self.MLP = nn.Sequential(*get_MLP_layers(n_neurons))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, xyzs, logits, hidden_states, i):
        if self.hidden_state_embedder is not None:
            hidden_states = hidden_states + self.hidden_state_embedder(i)
        outputs = self.MLP(torch.cat((self.xyz_embedder(xyzs), logits, hidden_states), dim=1))

        out_logits = outputs[:, 0:1] + logits
        out_hidden_states = outputs[:, 1:]
        return out_logits, out_hidden_states

    def init_weights(self):
        self.MLP.apply(initializer)


def _add_spherical(inputs):

    rho = torch.norm(inputs, p=2, dim=-1).view(-1, 1)

    theta = torch.acos(inputs[..., 2] / rho.view(-1)).view(-1, 1)
    theta[rho == 0] = 0

    phi = torch.atan2(inputs[..., 1], inputs[..., 0]).view(-1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * PI)
    phi[rho == 0] = 0

    return torch.cat([inputs, rho, theta, phi], dim=-1)


# https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L48
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] * (2 if self.kwargs['spherical_coordinates'] else 1)
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

    def forward(self, inputs):
        if self.kwargs['spherical_coordinates']:
            inputs = _add_spherical(inputs)
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class LearnableEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.cos_freq = nn.Parameter(2. ** torch.linspace(0., max_freq, steps=N_freqs))
            self.sin_freq = nn.Parameter(2. ** torch.linspace(0., max_freq, steps=N_freqs))
        else:
            self.cos_freq = nn.Parameter(torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs))
            self.sin_freq = nn.Parameter(torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs))

        self.out_dim = 2 * N_freqs * self.kwargs['input_dims'] + self.kwargs['input_dims']

    def forward(self, inputs):
        cos = torch.cos(inputs[..., None] * self.cos_freq).view((len(inputs), -1))
        sin = torch.sin(inputs[..., None] * self.sin_freq).view((len(inputs), -1))
        return torch.cat((inputs, cos, sin), -1)


def get_embedder(multires, use_spherical_coordinates=False, input_dims=3, include_input=True, disable_learnable=True):

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'spherical_coordinates': use_spherical_coordinates
    }

    return Embedder(**embed_kwargs) if disable_learnable else LearnableEmbedder(**embed_kwargs)
