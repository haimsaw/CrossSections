import json
from datetime import datetime
import numpy as np
from sampling.CSL import CSL
from sampling.Slicer import make_csl_from_mesh
from sampling.csl_to_xyz import csl_to_xyz

import argparse

def get_csl(bounding_planes_margin, save_path, name):
    # csl = CSL.from_csl_file(f"./data/csl-files/{name}.csl")

    # csl = CSL.from_csl_file(f"./data/csl_from_mesh/{name}_from_mesh.csl")

    csl = make_csl_from_mesh(f'./data/{name}.obj', save_path)

    csl.adjust_csl(bounding_planes_margin=bounding_planes_margin)
    return csl


parser = argparse.ArgumentParser(description='Run NeRP.')

parser.add_argument('gpu', type=int, default=0, help='an integer for the accumulator')


# sampling

# resolutions

# architecture
parser.add_argument('n_pe', type=int, default=4, dest='num_embedding_freqs', help='number of embedding freqs')
parser.add_argument('hss', type=int, default=32, dest='hidden_state_size', help='hidden state size')
parser.add_argument('n_loops', type=int, default=10, dest='n_loops', help='n of times we iterate')

# training

args = parser.parse_args()


class HP:
    def __init__(self):
        # sampling
        self.bounding_planes_margin = 0.05
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

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)