from CSL import CSL
import Renderer
from NaiveNetwork import *
from Mesher import marching_cubes


def main():
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

    bounding_planes_margin = 0.05
    sampling_margin = 0.2
    lr = 1e-2
    sampling_resolution_2d = (32, 32)
    sampling_resolution_3d = (100, 100, 100)
    epochs_list = [25, 25, 50, 50, 100]

    csl.adjust_csl(bounding_planes_margin=bounding_planes_margin)

    network_manager = NetworkManager()
    network_manager.load_from_disk()

    '''
    network_manager.prepare_for_training(csl, sampling_resolution_2d, sampling_margin, lr)

    for i, epochs in enumerate(epochs_list):
        network_manager.train_network(epochs=epochs)
        if i < len(epochs_list) - 1:
            network_manager.refine_sampling()
        Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(50, 50, 50), model_alpha=0.05)
    
    Renderer.draw_scene_and_errors(network_manager, csl)
    '''

    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution_3d=(50, 50, 50), model_alpha=0.05)

    mesh = marching_cubes(network_manager, sampling_resolution=sampling_resolution_3d)
    Renderer.draw_mesh(mesh)


if __name__ == "__main__":
    main()

'''
todo:
	0.2 keep original proportions when resterizing
'''
