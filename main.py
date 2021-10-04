from CSL import CSL
import Renderer
from NaiveNetwork import *
from Mesher import marching_cubes


def main():
    # csl = CSL("csl-files/ParallelEight.csl")
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
    #margin = 0.05
    #csl.adjust_csl(margin=margin)
    sampling_resolution_3d = (64, 64, 64)
    sampling_resolution_2d = (32, 32)



    # Renderer.draw_rasterized_plane(csl.planes[3], resolution=(256, 256), margin=0.2)
    # Renderer.draw_plane(csl.planes[27])

    # Renderer.draw_scene(csl)
    # Renderer.draw_rasterized_scene_cells(csl, sampling_resolution=sampling_resolution_2d, margin=margin)

    network_manager = NetworkManager()
    network_manager.load_from_disk()
    mesh = marching_cubes(network_manager, sampling_resolution=sampling_resolution_3d)
    Renderer.show_mesh(mesh)
    mesh.save('mesh.stl')

    # Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=sampling_resolution_3d, model_alpha=0.05)
    # Renderer.draw_rasterized_scene_cells(csl, sampling_resolution=sampling_resolution_3d, margin=margin, show_empty_planes=True)

    '''
    # network_manager.prepare_for_training(csl, lr=1e-2, sampling_resolution=(32, 32))
    # Renderer.draw_dataset(network_manager.dataset)

    network_manager.train_network(epochs=1)
    network_manager.refine_sampling()
    Renderer.draw_model(network_manager, sampling_resolution=(64, 64, 64))
    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(64, 64, 64))
    Renderer.draw_dataset(network_manager.dataset)

    network_manager.train_network(epochs=25)
    network_manager.refine_sampling()
    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(64, 64, 64))
    network_manager.save_to_disk()

    network_manager.train_network(epochs=25)
    network_manager.refine_sampling()
    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(64, 64, 64))
    network_manager.save_to_disk()

    network_manager.train_network(epochs=25)
    network_manager.refine_sampling()
    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(64, 64, 64))
    network_manager.save_to_disk()

    network_manager.show_train_losses()

    '''


if __name__ == "__main__":
    main()

'''
todo:
	0.2 keep original proportions when resterizing
'''
