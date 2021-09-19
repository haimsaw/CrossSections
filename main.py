from CSL import CSL
import Renderer
from NaiveNetwork import *


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

    csl.adjust_csl(margin=0.2)

    # Renderer.draw_rasterized_plane(csl.planes[3], resolution=(256, 256), margin=0.2)
    # Renderer.show_plane(csl.planes[3])

    # Renderer.draw_scene(csl, box)
    # Renderer.draw_rasterized_scene_cells(csl, sampling_resolution=(32, 32), margin=0.2)

    network_manager = NetworkManager()
    network_manager.prepare_for_training(csl, lr=1e-2, sampling_resolution=(34, 34))
    Renderer.draw_rasterized_scene_cells(csl, sampling_resolution=(10, 10), margin=0.2)

    network_manager.train_network(epochs=1)
    network_manager.refine_sampling()
    Renderer.draw_model_and_scene(network_manager, csl, sampling_resolution=(64, 64, 64))

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

    # renderer = Renderer.Renderer(csl, box)
    # renderer.event_loop()


if __name__ == "__main__":
    main()

'''
todo:
	0.2 keep original proportions when resterizing
	visualization - show with original planes
	    axis equal - show correct proprtions
	reduce 2d sample 32*32 and over sample on errors (see chat)
'''
