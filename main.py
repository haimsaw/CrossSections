from CSL import *
from Renderer import Renderer3D
from NetManager import *
from Mesher import *
from Helpers import *
from Modules import *
from stl import mesh as mesh2


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


def main():
    bounding_planes_margin = 0.05
    sampling_margin = 0.5
    lr = 1e-2
    root_sampling_resolution_2d = (30, 30)
    l1_sampling_reolution_2d = (30, 30)
    sampling_resolution_3d = (30, 30, 30)
    hidden_layers = [16, 32, 32, 32]
    n_epochs = 1
    embedder = get_embedder(10)

    csl = get_csl(bounding_planes_margin)

    # csl. planes = [csl.planes[0]]

    renderer = Renderer3D()
    renderer.add_scene(csl)
    #renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    # renderer.add_rasterized_scene(csl, root_sampling_resolution_2d, sampling_margin, show_empty_planes=False, show_outside_shape=True)
    renderer.show()
    return


    network_manager_root = HaimNetManager(csl, hidden_layers, embedder)
    # network_manager_root.load_from_disk()
    network_manager_root.prepare_for_training(root_sampling_resolution_2d, sampling_margin, lr)
    network_manager_root.train_network(epochs=n_epochs)

    mesh = marching_cubes(network_manager_root, sampling_resolution_3d)
    renderer = Renderer3D()
    renderer.add_mesh(mesh)
    renderer.add_scene(csl)
    renderer.add_model_errors(network_manager_root)
    renderer.show()

    network_manager_root.requires_grad_(False)

    octnetree_manager_l1 = OctnetreeManager(csl, hidden_layers, network_manager_root, embedder)
    octnetree_manager_l1.prepare_for_training(l1_sampling_reolution_2d, sampling_margin, lr)
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
    should sample from [-1, 1] in all directions

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
