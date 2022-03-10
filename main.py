from datetime import datetime
import json

from CSL import *
from ContourDataset import ContourDataset, ContourDatasetFake
from SlicesDataset import SlicesDataset, SlicesDatasetFake
from Renderer import *
from Mesher import *
from OctnetTree import *


def get_csl(bounding_planes_margin):
    csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/Heart-25-even-better.csl")
    # csl = CSL("csl-files/Armadillo-23-better.csl")
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
        self.root_sampling_resolution_2d = (32, 32)
        self.sampling_resolution_3d = (64, 64, 64)
        self.contour_sampling_resolution = 5

        # architecture
        self.num_embedding_freqs = 4
        self.hidden_layers = [128]*6  # [64, 64, 64, 64, 64]
        self.is_siren = False

        # loss
        self.density_lambda = 1

        # vals constraints
        self.contour_val_lambda = 0

        self.inter_lambda = 0# 1e0
        self.inter_alpha = -1e2

        self.off_surface_lambda = 0
        self.off_surface_epsilon = 1e-3

        # grad constraints
        self.eikonal_lambda = 0# 1e-3

        self.contour_normal_lambda = 0# 1e-1
        self.contour_tangent_lambda = 1 # 1e-1

        # training
        self.weight_decay = 1e-3  # l2 regularization
        self.epochs = 2
        self.scheduler_step = 5
        self.lr = 1e-2

        # inference
        self.sigmoid_on_inference = False

        self.now = str(datetime.now())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def main():
    hp = HP()

    csl = get_csl(hp.bounding_planes_margin)

    print(f'loss: density={hp.density_lambda}, eikonal={hp.eikonal_lambda}, contour_val={hp.contour_val_lambda}, contour_normal={hp.contour_normal_lambda}, contour_tangent={hp.contour_tangent_lambda} inter={hp.inter_lambda} off={hp.off_surface_lambda}')

    tree = OctnetTree(csl, hp.oct_overlap_margin, hp.hidden_layers, get_embedder(hp.num_embedding_freqs), hp.is_siren)

    # d2_res = [i * (2 ** (tree.depth + 1)) for i in hp.root_sampling_resolution_2d]
    should_calc_density = hp.density_lambda > 0 or hp.inter_lambda > 0

    slices_dataset = SlicesDataset(csl, sampling_resolution=hp.root_sampling_resolution_2d, sampling_margin=hp.sampling_margin, should_calc_density=should_calc_density)
    contour_dataset = ContourDataset(csl, round(len(slices_dataset) / len(csl)))  # todo haim

    print(f'slices={len(slices_dataset)}, contour={len(contour_dataset)}, samples_per_edge={round(len(slices_dataset) / len(csl))}')

    # level 0:
    tree.prepare_for_training(slices_dataset, contour_dataset, hp)
    tree.train_network(epochs=hp.epochs)

    # tree.show_train_losses()

    try:
        #mesh_mc = marching_cubes(tree, hp.sampling_resolution_3d, hp.sigmoid_on_inference)
        #mesh_mc.save('./artifacts/output_mc.stl')

        mesh_dc = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=True, use_sigmoid=False)
        mesh_dc.save('./artifacts/output_dc_grad.obj')

        mesh_dc = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=False, use_sigmoid=False)
        mesh_dc.save('./artifacts/output_dc_no_grad.obj')

        renderer = Renderer3D()
        renderer.add_scene(csl)
        renderer.add_mesh(mesh_mc)
        # renderer.add_model_grads(tree, get_xyzs_in_octant(None, sampling_resolution_3d=(10, 10, 10)))
        renderer.show()

    except ValueError as e:
        print(e)
    finally:
        for dim in (0, 2):
            for dist in np.linspace(-1, 1, 3):
                renderer = Renderer2D()
                renderer.heatmap([100] * 2, tree, dim, dist, True, hp.sigmoid_on_inference)
                renderer.save('./artifacts/')
                renderer.clear()


if __name__ == "__main__":
    main()

'''
todo
check if tree is helping or its just capacity 
delete INetManager

read: 
https://arxiv.org/abs/2202.01999 - nural dc
https://arxiv.org/pdf/1807.02811.pdf - Bayesian Optimization

create slicer for chamfer compare
track experiments with mlflow?
serialize a tree (in case collab crashes)
increase sampling (in prev work he used 2d= 216, 3d=300)

Sheared weights / find a way to use symmetries 
reduce capacity of lower levels (make #params in each level equal)

smooth loss - sink horn 

Use sinusoidal activations (SIREN/SAPE)
Scale & translate each octant to fit [-1,1]^3

Use loss from the upper level to determine depth \ #epochs

nerfs literature review  
'''
