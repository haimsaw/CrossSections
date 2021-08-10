from CSL import CSL
from Renderer import Renderer2 #, Renderer
from NaiveNetwork import *


def main():
    csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")
    # csl = CSL("csl-files/SideBishop.csl")

    # csl = CSL("csl-files/Heart-25-even-better.csl")

    # csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/rocker-arm.csl")

    # csl = CSL("csl-files/Brain.csl")

    csl.centralize()
    csl.rotate_by_pca()  # todo not rotating plane coordinates
    csl.scale()
    box = csl.add_boundary_planes(margin=0.2)  # todo show and sample empty planes

    # csl.planes[27].get_pca_projected_plane().show_rasterized(resolution=(256, 256), margin=0.2)
    # csl.planes[27].get_pca_projected_plane().show_plane()

    # Renderer2().draw_scene(csl, box)
    # Renderer2().draw_rasterized_scene(csl, box, sampling_resolution=(256, 256), margin=0.2)

    network_manager = NetworkManager().prepere_for_training(csl, lr=1e-2)
    network_manager.train_network(epochs=50)
    network_manager.show_train_losses()

    Renderer2().draw_model(network_manager, sampling_resolution=(64, 64, 64))
    x = 1
    # renderer = Renderer(csl, box)
    # renderer.event_loop()


if __name__ == "__main__":
    main()

'''
todo:
	0.2 keep original proportions when resterizing
	0.3 show empty slices
	0.4 train basic network with 5 FC layars * 256 nurons -> test by eye and in between slices 
'''
