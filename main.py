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

    sampling_resolution_3d = (40, 40, 40)
    hidden_layers = [16, 32, 32, 32]
    epochs = 0
    embedder = get_embedder(4)
    scheduler_step = 5
    oct_overlap_margin = 0.25  # should be 1/2^k

    csl = get_csl(bounding_planes_margin)

    '''
    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    #renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    renderer.add_rasterized_scene(csl, root_sampling_resolution_2d, sampling_margin, show_empty_planes=True, show_outside_shape=True)
    renderer.show()
    '''

    xyzs_vertex = get_xyzs_in_octant(np.array([[0.25]*3, [-0.25]*3]), sampling_resolution_3d)
    xyzs_edge = get_xyzs_in_octant(np.array([[0.25, 0.25, 0.75], [-0.25, -0.25, -0.75]]), sampling_resolution_3d)
    xyzs_no_boundary = get_xyzs_in_octant(np.array([[0.75]*3, [-0.75]*3]), sampling_resolution_3d)
    xyzs_no_boundary_l2 = get_xyzs_in_octant(np.array([[0.875]*3, [-0.875]*3]), sampling_resolution_3d)
    xyzs_small_depth2 = get_xyzs_in_octant(np.array([[0.55]*3, [0.45]*3]), sampling_resolution_3d)
    xyzs_few_planes = get_xyzs_in_octant([[1.0, 1.0, 0], [-1.0, -1.0, -1]], (50, 50, 4))
    xyzs_one_plane = get_xyzs_in_octant([[1.0, 1.0, -0.25], [-1.0, -1.0, -0.25]], (40, 40, 1))

    xyzs_all = get_xyzs_in_octant(None, sampling_resolution_3d)

    tree = OctnetTree(csl, oct_overlap_margin, hidden_layers, embedder)

    # todo root_sampling_resolution_2d
    # level 0:
    dataset = RasterizedCslDataset(csl, sampling_resolution=root_sampling_resolution_2d, sampling_margin=sampling_margin,
                                   target_transform=torch.tensor, transform=torch.tensor)

    tree.prepare_for_training(dataset, lr, scheduler_step)
    tree.train_network(epochs=epochs)

    # draw_blending_errors(tree, xyzs_all, f'{tree.depth} xyzs_all ')

    # level 1
    tree.prepare_for_training(dataset, lr, scheduler_step)
    tree.train_network(epochs=epochs)

    #draw_blending_errors(tree, xyzs_vertex)
    #draw_blending_errors(tree, xyzs_edge)
    draw_blending_errors(tree, xyzs_one_plane, f'{tree.depth} xyzs_one_plane ')

    draw_blending_errors(tree, xyzs_few_planes, f'{tree.depth} xyzs_few_planes ')
    draw_blending_errors(tree, xyzs_no_boundary, f'{tree.depth} xyzs_no_boundary ')
    draw_blending_errors(tree, xyzs_all, f'{tree.depth} xyzs_all ')

    # level 3
    tree.prepare_for_training(dataset, lr, scheduler_step)
    tree.train_network(epochs=epochs)

    draw_blending_errors(tree, xyzs_no_boundary_l2, f'{tree.depth} xyzs_no_boundary_l2 ')
    draw_blending_errors(tree, xyzs_all, f'{tree.depth} xyzs_all ')



    # mesh = marching_cubes(network_manager_root, sampling_resolution_3d)
    # renderer = Renderer3D()
    # renderer.add_mesh(mesh)
    # renderer.add_scene(csl)
    # renderer.add_model_errors(network_manager_root)
    # renderer.show()


def draw_blending_errors(tree, xyzs, title):
    labels = tree.soft_predict(xyzs)
    print(f'max={max(labels)}, min={min(labels)}, n={len(labels)} depth={tree.depth}')

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_title(title)

    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1, 1)
    #ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.scatter(*xyzs[(labels != 1) & (labels != 0)].T, c=labels[(labels != 1) & (labels != 0)], alpha=0.2)
    ax.scatter(*xyzs.T, c=labels, alpha=0.2)
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

'''
todo
check if tree is helping or its just capacity 

Use sinusoidal activations (SIREN)
Scale & translate each octant to fit [-1,1]^3
Sheared weights / find a way to use symmetries 
Use loss from the upper level to determine depth \#epochs

'''