import numpy as np

from CSL import *
from Renderer import Renderer3D
from NetManager import *
from Mesher import *
from Helpers import *
from Modules import *
from stl import mesh as mesh2
from Octree2 import *


def main():
    bounding_planes_margin = 0.05
    sampling_margin = bounding_planes_margin
    lr = 1e-2
    root_sampling_resolution_2d = (30, 30)
    l1_sampling_resolution_2d = (30, 30)
    l2_sampling_resolution_2d = (30, 30)

    sampling_resolution_3d = (50, 50, 50)
    hidden_layers = [16, 32, 32, 32]
    epochs = 0
    embedder = get_embedder(4)
    scheduler_step = 5
    oct_overlap_margin = 0.2

    csl = get_csl(bounding_planes_margin)

    '''
    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    #renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    renderer.add_rasterized_scene(csl, root_sampling_resolution_2d, sampling_margin, show_empty_planes=True, show_outside_shape=True)
    renderer.show()
    '''

    #xyzs = get_xyzs_in_octant(np.array([[0.2]*3, [-.2]*3]), sampling_resolution_3d)
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)
    tree = OctnetTree(csl, oct_overlap_margin, hidden_layers, embedder)


    # todo root_sampling_resolution_2d
    # todo put in loop
    dataset = RasterizedCslDataset(csl, sampling_resolution=root_sampling_resolution_2d, sampling_margin=sampling_margin,
                                   target_transform=torch.tensor, transform=torch.tensor)
    for _ in range(3):

        tree.prepare_for_training(dataset, lr, scheduler_step)
        tree.train_network(epochs=epochs)
        draw_blending_errors(tree, xyzs)

    # mesh = marching_cubes(network_manager_root, sampling_resolution_3d)
    # renderer = Renderer3D()
    # renderer.add_mesh(mesh)
    # renderer.add_scene(csl)
    # renderer.add_model_errors(network_manager_root)
    # renderer.show()


def draw_blending_errors(tree, xyzs):
    labels = tree.soft_predict(xyzs)
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(*xyzs[(labels != 1) & (labels != 0)].T, c=labels[(labels != 1) & (labels != 0)], alpha=0.01)
    plt.show()


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

if __name__ == "__main__":
    main()