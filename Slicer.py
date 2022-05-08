import numpy as np

from CSL import *
from Renderer import Renderer3D, Renderer2D
from meshcut import cross_section
import pywavefront


def make_csl_from_mesh(filename):
    n_slices = 20
    plane_origins = [(0, 0, d) for d in np.linspace(-1, 1, n_slices)]
    plane_normals = [(0, 0, 1.0)] * n_slices

    # todo make sure the normals are normlized
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    assert len(scene.mesh_list) == 1

    verts = np.array(scene.vertices)
    verts -= np.mean(verts,  axis=0)
    verts /= np.max(np.absolute(verts))

    faces = scene.mesh_list[0].faces
    model_name = filename.split('/')[-1].split('.')[0]

    csl = CSL.from_mesh(model_name, plane_origins,  plane_normals, verts, faces)
    return csl


if __name__ == '__main__':
    make_csl_from_mesh()
