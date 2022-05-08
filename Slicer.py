import numpy as np

from CSL import *
from Renderer import Renderer3D, Renderer2D
from meshcut import cross_section
import pywavefront


def make_csl_from_mesh(filename):
    n_slices = 50
    plane_origins = [(0, 0, d) for d in np.linspace(-1, 1, n_slices)]
    plane_normals = [(0, 0, 1.0)] * n_slices
    # todo make sure the normals are normlized

    csl = CSL.from_mesh(filename, plane_origins,  plane_normals)
    return csl


if __name__ == '__main__':
    make_csl_from_mesh()
