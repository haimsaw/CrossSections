import numpy as np

import CSL
from meshcut import cross_section
import pywavefront


def make_csl():
    scene = pywavefront.Wavefront('./mesh/armadillo.obj', collect_faces=True)


    plane_origins =[(0, 0.30, 0), (0, -0.30, 0), (0, 0, 0)]
    plane_normals = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
    for origin, normal in zip(plane_origins, plane_normals):
        intersection = cross_section(scene.vertices, scene.mesh_list[0].faces, plane_orig=origin, plane_normal=normal)



if __name__ == '__main__':
    make_csl()