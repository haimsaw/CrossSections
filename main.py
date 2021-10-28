from CSL import *
from Renderer import Renderer3D
from NetManager import *
from Mesher import marching_cubes
from Helpers import *


def main():
    bounding_planes_margin = 0.05
    sampling_margin = 0.5
    lr = 1e-2
    root_sampling_resolution_2d = (32, 32)
    l1_sampling_resolution_2d = (64, 64)
    layers = (3, 16, 32, 32, 64, 1)
    n_epochs = 5

    csl = get_csl(bounding_planes_margin)

    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_rasterized_scene(csl, sampling_resolution_2d, sampling_margin, show_empty_planes=False, show_outside_shape=True)
    # renderer.show()

    network_manager_root = HaimNetManager(layers)
    # network_manager_root.load_from_disk()
    network_manager_root.prepare_for_training(csl, root_sampling_resolution_2d, sampling_margin, lr, octant=None)
    network_manager_root.train_network(epochs=n_epochs)

    network_manager_root.requires_grad_(False)

    network_manager_root.requires_grad_(False)
    network_manager_layer1 = [HaimNetManager(layers, residual_module=network_manager_root.module) for _ in range(8)]
    octanes = get_octets(*add_margin(*get_top_bottom(csl.all_vertices), sampling_margin))

    for network_manager, octant in zip(network_manager_layer1, octanes):
        network_manager.prepare_for_training(csl, l1_sampling_resolution_2d, sampling_margin, lr, octant=octant)
        network_manager.train_network(epochs=n_epochs)
        #network_manager.show_train_losses()


    # Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(50, 50, 50), model_alpha=0.05)

    # mesh = marching_cubes(network_manager, sampling_resolution=sampling_resolution_3d)
    # mesh.save('mesh.stl')


if __name__ == "__main__":
    main()

'''
todo:
    learn residue 

    use k3d for rendering in collab https://github.com/K3D-tools/K3D-jupyter/blob/main/HOW-TO.md
    CGAL 5.3 - 3D Surface Mesh Generation https://doc.cgal.org/latest/Surface_mesher/index.html
	0.2 keep original proportions when resterizing
'''
