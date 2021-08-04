from CSL import CSL
from Renderer import Renderer2 #Renderer
import torch


def main():
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")


    #csl = CSL("csl-files/Heart-25-even-better.csl")

    csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/rocker-arm.csl")

    # csl = CSL("csl-files/Brain.csl")

    csl.centralize()
    csl.rotate_by_pca() # todo not rotating plane coordinates

    box = csl.add_boundary_planes(margin=0.2)

    # csl.planes[27].get_pca_projected_plane().show_rasterized(resolution=(256, 256), margin=0.2)
    # csl.planes[27].get_pca_projected_plane().show_plane()

    Renderer2(csl, box).draw_scene()
    #Renderer2(csl, box).draw_rasterized_scene(resolution=(256, 256, 256), margin=0.2)


    #renderer = Renderer(csl, box)
    #renderer.event_loop(is_resterized=False)


if __name__ == "__main__":
    main()

'''
todo:
	0.1 draw rastarization in 3d
	0.2 keep original proportions when resterizing
	0.3 show empty slices
	0.4 train basic network with 5 FC layars * 256 nurons -> test by eye and in between slices 
'''
