import json
from datetime import datetime
import numpy as np
from sampling.CSL import CSL
from sampling.Slicer import make_csl_from_mesh
from sampling.csl_to_xyz import csl_to_xyz

import argparse

from_mesh_models = ['armadillo', 'lamp004_fixed', 'eight_15', 'eight_20', 'astronaut']

parser = argparse.ArgumentParser(description='Run NeRP.')

parser.add_argument('out_dir', type=str, required=True, help='out dir to save artifacts')
parser.add_argument('--shape', type=str, dest='model_name', default='all', choices=['all']+from_mesh_models, help='model name to run')

parser.add_argument('--gpu', type=int, default=0, help='an integer for the accumulator')


# sampling
parser.add_argument('--bounding_planes_margin', type=float, default=0.05, dest='bounding_planes_margin', help='the margin of bbox')


# resolutions

# architecture
parser.add_argument('--n_pe', type=int, default=4, dest='num_embedding_freqs', help='number of embedding freqs')
parser.add_argument('--hss', type=int, default=32, dest='hidden_state_size', help='hidden state size')
parser.add_argument('--n_loops', type=int, default=10, dest='n_loops', help='n of times we iterate')

# training

args = parser.parse_known_args()


class HP:
    def __init__(self):
        # sampling
        self.sampling_margin = 0.05  # same as bounding_planes_margin
        self.sampling_radius = [(1/2)**4, (1/2)**5, (1/2)**6, (1/2)**7, (1/2)**8, (1/2)**9]
        self.n_samples = [2, 2, 3, 3, 4, 5]
        self.n_white_noise = 128

        # resolutions
        self.root_sampling_resolution_2d = (64, 64)
        self.sampling_resolution_3d = (128, 128, 128)
        self.intermediate_sampling_resolution_3d = (64, 64, 64)

        # architecture
        self.spherical_coordinates = False
        self.hidden_layers = [64]*5
        self.hidden_state_size = 32
        self.hidden_state_embedder = True

        # loss
        self.density_lambda = 1

        # training
        self.weight_decay = 1e-3  # l2 regularization
        self.epochs_batches = [20, 50, 50, 100, 100, 100]
        self.scheduler_step = 10
        self.scheduler_gamma = 0.9
        self.lr = 1e-2
        self.batch_size = 2 ** 13

        assert len(self.epochs_batches) <= len(self.sampling_radius) and len(self.epochs_batches) <= len(self.n_samples)

        self.now = str(datetime.now())
