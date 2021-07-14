from CSL import CSL
from Renderer import Renderer


def main():
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")

    # csl = CSL("csl-files/Heart-25-even-better.csl")

    #csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    csl = CSL("csl-files/rocker-arm.csl")

    #csl = CSL("csl-files/Brain.csl")

    csl.centralize()
    box = csl.add_boundary_planes(margin=0.2)

    csl.planes[30].get_pca_projected_plane().show_rasterized(shape=(256, 256), margin=0.2)

    csl.planes[30].get_pca_projected_plane().show_plane()

    #renderer = Renderer(csl, box)
    #renderer.event_loop()


if __name__ == "__main__":
    main()

'''
todo:
	0? draw the shape filled in the csl visualization()
	3. rasterize the plane:
		3.a pca on the points fox axis, origin in mean to get params for the plane
		3.b take -+20% of empty space
		3.c get color for reach pixel (256*256 pixels in each direction)
	4. visualize raster of the planes
'''
