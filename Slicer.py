import numpy as np

from CSL import *
from Renderer import Renderer3D, Renderer2D
from meshcut import cross_section
import pywavefront


def make_csl():
    filename = './mesh/armadillo.obj'
    plane_origins = [(0, 0.30, 0), (0, -0.30, 0), (0, 0, 0)]
    plane_normals = [(0, 1.0, 0), (1.0, 0, 0), (0, 0, 1.0)]

    csl = CSL.from_mesh(filename, plane_origins,  plane_normals)

    r = Renderer3D()
    r.add_scene(csl)
    r.show()

    for p in csl.planes:
        r = Renderer2D()
        r.draw_plane(p)
        r.show()

if __name__ == '__main__':
    make_csl()