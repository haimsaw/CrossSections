from CSL import CSL
import Renderer
from NaiveNetwork import *


def main():
    csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")
    # csl = CSL("csl-files/SideBishop.csl")

    # csl = CSL("csl-files/Heart-25-even-better.csl")

    #csl = CSL("csl-files/Armadillo-23-better.csl")
    # csl = CSL("csl-files/Horsers.csl")

    # csl = CSL("csl-files/rocker-arm.csl")

    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/Skull-20.csl")

    # csl = CSL("csl-files/Brain.csl")

    csl.centralize()
    csl.rotate_by_pca()
    csl.scale()
    box = csl.add_boundary_planes(margin=0.2)  # todo show and sample empty planes

    # Renderer.draw_rasterized_plane(csl.planes[3], resolution=(256, 256), margin=0.2)
    # Renderer.show_plane(csl.planes[3])

    # Renderer.draw_scene(csl, box)
    Renderer.draw_rasterized_scene(csl, sampling_resolution=(32, 32), margin=0.2)

    network_manager = NetworkManager()
    # network_manager.load_from_disk()
    # network_manager.prepere_for_training(csl, lr=1e-2)
    # network_manager.train_network(epochs=1)
    # network_manager.show_train_losses()

    # Renderer.draw_model(network_manager, sampling_resolution=(64, 64, 64))

    # renderer = Renderer.Renderer(csl, box)
    # renderer.event_loop()


if __name__ == "__main__":
    main()

'''
todo:
	0.2 keep original proportions when resterizing
	0.3 show empty slices
	visualization - show with original planes
	    axis equal - show correct proprtions
	reduce 2d sample 32*32 and over sample on errors (see chat)
'''
