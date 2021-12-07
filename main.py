from CSL import *
from Renderer import Renderer3D
from NetManager import *
from Mesher import *
from Helpers import *
from Modules import *
from stl import mesh as mesh2
from Octree2 import *


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


def main():
    bounding_planes_margin = 0.05
    sampling_margin = bounding_planes_margin
    lr = 1e-2
    root_sampling_resolution_2d = (30, 30)
    l1_sampling_reolution_2d = (30, 30)
    sampling_resolution_3d = (30, 30, 30)
    hidden_layers = [16, 32, 32, 32]
    n_epochs = 1
    embedder = get_embedder(10)
    scheduler_step = 5
    oct_overlap_margin = 0.2

    csl = get_csl(bounding_planes_margin)

    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    #renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    renderer.add_rasterized_scene(csl, root_sampling_resolution_2d, sampling_margin, show_empty_planes=True, show_outside_shape=True)
    renderer.show()


    tree = OctnetTree(csl, oct_overlap_margin, hidden_layers, embedder)
    tree.train_leaves(sampling_resolution=root_sampling_resolution_2d, sampling_margin=sampling_margin, lr=lr, scheduler_step=scheduler_step, n_epochs=n_epochs)
    '''
    mesh = marching_cubes(network_manager_root, sampling_resolution_3d)
    renderer = Renderer3D()
    renderer.add_mesh(mesh)
    renderer.add_scene(csl)
    renderer.add_model_errors(network_manager_root)
    renderer.show()
    '''

    tree.add_level()
    tree.train_leaves(sampling_resolution=l1_sampling_reolution_2d, sampling_margin=sampling_margin, lr=lr, scheduler_step=scheduler_step, n_epochs=n_epochs)

    #tree.add_level()
    #tree.train_leaves(sampling_resolution=l1_sampling_reolution_2d, sampling_margin=sampling_margin, lr=lr, scheduler_step=scheduler_step, n_epochs=n_epochs)

    # Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(50, 50, 50), model_alpha=0.05)

    mesh = marching_cubes(tree, sampling_resolution_3d)
    renderer = Renderer3D()
    renderer.add_mesh(mesh)
    renderer.show()
    # mesh.save('mesh.stl')


if __name__ == "__main__":
    main()