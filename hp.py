import json
import sys
from datetime import datetime
import numpy as np


import argparse

from_mesh_models = ['armadillo', 'lamp004_fixed', 'eight_15', 'eight_20',
                    'balloondog', 'FlexiRex', 'dice', 'Elephant', 'astronaut']
mri_models = ['Heart-25-even-better', 'Vetebrae', 'Skull-20', 'Abdomen', 'Brain']

parser = argparse.ArgumentParser(description='Run NeRP.')

# run
parser.add_argument('out_dir', type=str, help='out dir to save artifacts')
parser.add_argument('--model', type=str, dest='model_name', default='all', choices=['all']+from_mesh_models, help='model name to run')
parser.add_argument('-run_mri', action='store_true',  help='run all mri models (ignores --model)')

parser.add_argument('--gpu', type=int, default=-1, help='which gpu device to use')
parser.add_argument('--seed', type=int, default=11, help='random seed to use (-1 for random)')


# sampling
parser.add_argument('--bounding_planes_margin', type=float, default=0.05, dest='bounding_planes_margin', help='the margin of bbox')
parser.add_argument('--nrd', type=int, default=3, dest='n_refined_datasets', help='n of refined datasets to use (none for all)')
parser.add_argument('-perturb', action='store_true', dest='should_perturb_samples', help='perturb samples')
parser.add_argument('--n_white_noise', type=int, default=1024, dest='n_white_noise', help='n of random points to sample at each plane')
parser.add_argument('-no_refine', action='store_true',  help='run all datadets')

# resolutions

# architecture
parser.add_argument('--num_embedding_freqs', type=int, default=4, dest='num_embedding_freqs', help='number of embedding freqs')
parser.add_argument('--hidden_state_size', type=int, default=32, dest='hidden_state_size', help='hidden state size')
parser.add_argument('--n_loops', type=int, default=10, dest='n_loops', help='n of times we iterate')
parser.add_argument('-disable_learnable_embed', action='store_true', help='learn embedded freqs')

# training
parser.add_argument('--n_samples', nargs='*', type=int, default=[2, 2, 3, 3, 4, 5])
parser.add_argument('--sampling_radius_exp', nargs='*', type=int, default=[4, 5, 6, 7, 8, 9], help='the exp of sampling radius (base 0.5)')

parser.add_argument('--scheduler_step', type=int, default=10, help='in how many iterations should we reduce lr')

if sys.argv[1] != '--mode=client':
    args = parser.parse_args()
else:
    args = parser.parse_args(['null'])


class HP:
    def __init__(self):
        # sampling
        self.sampling_margin = 0.05  # same as bounding_planes_margin

        # resolutions
        self.root_sampling_resolution_2d = (64, 64)
        self.sampling_resolution_3d = (300, 300, 300)
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
        self.scheduler_gamma = 0.9
        self.lr = 1e-2
        self.batch_size = 2 ** 13

        assert len(self.epochs_batches) <= len(args.sampling_radius_exp) and len(self.epochs_batches) <= len(args.n_samples)

        self.now = str(datetime.now())
