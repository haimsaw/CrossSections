from CSL import CSL
from Renderer import Renderer3D
from NetManager import *
from Mesher import marching_cubes
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
    layers = (3, 128, 256, 512, 512, 1)
    n_epochs = 1

    csl = get_csl(bounding_planes_margin)

    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_rasterized_scene(csl, sampling_resolution_2d, sampling_margin, show_empty_planes=False, show_outside_shape=True)
    renderer.show()

    network_manager_root = HaimNetManager(layers)
    # network_manager_root.load_from_disk()
    network_manager_root.prepare_for_training(csl, root_sampling_resolution_2d, sampling_margin, lr, octant=None)
    network_manager_root.train_network(epochs=n_epochs)

    network_manager_layer1 = [HaimNetManager(layers) for _ in range(8)]
    octanes = get_octets(*add_margin(*get_top_bottom(csl.all_vertices), sampling_margin))

    def run_sub_net(network_manager, octant):
        network_manager.prepare_for_training(csl, l1_sampling_resolution_2d, sampling_margin, lr, octant=octant)
        network_manager.train_network(epochs=n_epochs)

    map(lambda network, octant: run_sub_net(network, octant), zip(network_manager_layer1, octanes))

    # Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(50, 50, 50), model_alpha=0.05)

    # mesh = marching_cubes(network_manager, sampling_resolution=sampling_resolution_3d)
    # mesh.save('mesh.stl')


if __name__ == "__main__":
    main()

'''
todo:
    scale csl so that bounding box is at 1,0,0
    genarate all octants


    use k3d for rendering in collab https://github.com/K3D-tools/K3D-jupyter/blob/main/HOW-TO.md
    CGAL 5.3 - 3D Surface Mesh Generation https://doc.cgal.org/latest/Surface_mesher/index.html
	0.2 keep original proportions when resterizing
'''
