from CSL import *
from Renderer import Renderer3D
from NetManager import *
from Mesher import *
from Helpers import *


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
    sampling_margin = 0.5
    lr = 1e-2
    root_sampling_resolution_2d = (32, 32)
    l1_sampling_resolution_2d = (64, 64)
    sampling_resolution_3d = (50, 50, 50)

    layers = (3, 16, 32, 32, 64, 1)
    layers = (3, 16, 1)

    n_epochs = 1

    csl = get_csl(bounding_planes_margin)

    # renderer = Renderer3D()
    # renderer.add_scene(csl)
    # renderer.add_rasterized_scene(csl, sampling_resolution_2d, sampling_margin, show_empty_planes=False, show_outside_shape=True)
    # renderer.show()

    network_manager_root = HaimNetManager(csl, layers)
    # network_manager_root.load_from_disk()
    network_manager_root.prepare_for_training(root_sampling_resolution_2d, sampling_margin, lr)
    network_manager_root.train_network(epochs=n_epochs)

    network_manager_root.requires_grad_(False)

    octnetree_manager_l1 = OctnetreeManager(csl, layers, network_manager_root)
    octnetree_manager_l1.prepare_for_training(l1_sampling_resolution_2d, sampling_margin, lr)
    octnetree_manager_l1.train_network(epochs=n_epochs)

    # Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(50, 50, 50), model_alpha=0.05)

    mesh = marching_cubes(octnetree_manager_l1, sampling_resolution_3d)
    renderer = Renderer3D()
    renderer.add_mesh(mesh)
    renderer.show()
    # mesh.save('mesh.stl')


if __name__ == "__main__":
    main()

'''
todo:
    check if i should scale everything to 1?
    check if i should use softmax when predicting
    
    add scene to mesh

    positional encoding
    overlapping octanes
    make octnetree_manager as actual octree
    keep original proportions when resterizing

    
    use k3d for rendering in collab https://github.com/K3D-tools/K3D-jupyter/blob/main/HOW-TO.md
    CGAL 5.3 - 3D Surface Mesh Generation https://doc.cgal.org/latest/Surface_mesher/index.html
'''
