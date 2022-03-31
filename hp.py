import json
from datetime import datetime

from CSL import CSL

INSIDE_LABEL = 0.0
OUTSIDE_LABEL = 1.0


def get_csl(bounding_planes_margin):
    # csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/Heart-25-even-better.csl")
    csl = CSL("csl-files/Armadillo-23-better.csl")
    # csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/rocker-arm.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/Skull-20.csl")
    # csl = CSL("csl-files/Brain.csl")

    csl.adjust_csl(bounding_planes_margin=bounding_planes_margin)
    return csl


class HP:
    def __init__(self):
        # sampling
        self.bounding_planes_margin = 0.05
        self.sampling_margin = 0.05  # same as bounding_planes_margin
        self.oct_overlap_margin = 0.25

        # resolutions
        self.root_sampling_resolution_2d = (64, 64)
        self.sampling_resolution_3d = (64, 64, 64)
        self.n_samples_per_edge = 32

        # architecture
        self.num_embedding_freqs = 4
        self.spherical_coordinates = False
        self.hidden_layers = [64]*6
        self.depth = 2

        # loss
        self.initial_density_lambda = 1
        self.density_schedule_fraction = -1  # 3 / 4  # -1 for no schedule

        # vals constraints
        self.contour_val_lambda = 0

        self.inter_lambda = 0
        self.inter_alpha = 0

        self.off_surface_lambda = 0
        self.off_surface_epsilon = 0

        # grad constraints
        self.eikonal_lambda = 0
        self.zero_grad = 1e-4

        self.contour_normal_lambda = 1e-3
        self.contour_tangent_lambda = 1e-3

        # training
        self.weight_decay = 1e-3  # l2 regularization
        self.epochs = 50
        self.scheduler_step = 5
        self.scheduler_gamma = 0.9
        self.lr = 1e-3

        # inference
        self.sigmoid_on_inference = False

        self.now = str(datetime.now())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

