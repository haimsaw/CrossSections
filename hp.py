import json
from datetime import datetime
import numpy as np
from CSL import CSL
from Slicer import make_csl_from_mesh
from csl_to_xyz import csl_to_xyz


def get_csl(bounding_planes_margin, save_path, name):
    # csl = CSL.from_csl_file("data/csl-files/ParallelEight.csl")
    # csl = CSL.from_csl_file("data/csl-files/ParallelEightMore.csl")
    # csl = CSL.from_csl_file("data/csl-files/SideBishop.csl")
    # csl = CSL.from_csl_file("data/csl-files/Heart-25-even-better.csl")
    # csl = CSL.from_csl_file("data/csl-files/Armadillo-23-better.csl")
    # csl = CSL.from_csl_file("data/csl-files/Horsers.csl")
    # csl = CSL.from_csl_file("data/csl-files/rocker-arm.csl")
    # csl = CSL.from_csl_file("data/csl-files/Abdomen.csl")
    # csl = CSL.from_csl_file("data/csl-files/Vetebrae.csl")
    # csl = CSL.from_csl_file("data/csl-files/Skull-20.csl")
    # csl = CSL.from_csl_file("data/csl-files/Brain.csl")

    # csl = CSL.from_csl_file(f"./data/csl_from_mesh/{name}_from_mesh.csl")


    # csl = make_csl_from_mesh('./data/eight.obj', save_path)
    # csl = make_csl_from_mesh('data/armadillo.obj', save_path)
    csl = make_csl_from_mesh('data/obj/lamp004_fixed.obj', save_path)

    csl.adjust_csl(bounding_planes_margin=bounding_planes_margin)
    return csl


class HP:
    def __init__(self):
        # sampling
        self.bounding_planes_margin = 0.05
        self.sampling_margin = 0.05  # same as bounding_planes_margin
        self.oct_overlap_margin = 0.25
        self.refinement_type = 'edge'  # ['errors', 'edge', 'none']

        # resolutions
        self.root_sampling_resolution_2d = (64, 64)
        self.sampling_resolution_3d = (128, 128, 128)
        self.intermediate_sampling_resolution_3d = (64, 64, 64)

        # architecture
        self.num_embedding_freqs = 4
        self.spherical_coordinates = False
        self.hidden_layers = [64]*5
        self.hidden_state_size = 32
        self.hidden_state_embedder = True
        self.n_loops = 10

        # loss
        self.density_lambda = 1

        # training
        self.weight_decay = 1e-3  # l2 regularization
        self.epochs_batches = [50, 50, 100, 100, 100]
        self.scheduler_step = 10
        self.scheduler_gamma = 0.9
        self.lr = 1e-2
        self.batch_size = 2 ** 13

        self.now = str(datetime.now())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)