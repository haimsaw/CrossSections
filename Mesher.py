import numpy as np
from stl import mesh as mesh2
from Helpers import *
from skimage import measure


def marching_cubes(network_manager, sampling_resolution_3d, soft_predict=True):
    xyz = get_xyz_in_octant(None, sampling_resolution_3d)

    labels = (network_manager.soft_predict(xyz) if soft_predict else network_manager.hard_predict(xyz)).reshape(sampling_resolution_3d)

    vertices, faces, normals, values = measure.marching_cubes(labels, 0)

    my_mesh = mesh2.Mesh(np.zeros(faces.shape[0], dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = vertices[f[j], :]

    return my_mesh

