import numpy as np

from CSL import *
from Renderer import Renderer3D, Renderer2D
from meshcut import cross_section
import pywavefront


def make_csl(filename):
    n_slices = 50
    plane_origins = [(0, d, 0) for d in np.linspace(-1, 1, n_slices)]
    plane_normals = [(0, 1.0, 0)] * n_slices

    csl = CSL.from_mesh(filename, plane_origins,  plane_normals)
    csl.adjust_csl(0.05)

    r = Renderer3D()
    r.add_scene(csl)
    r.show()

    for p in csl.planes:
        r = Renderer2D()
        r.draw_plane(p)
        r.show()

if __name__ == '__main__':
    make_csl()