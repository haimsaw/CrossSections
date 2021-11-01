import numpy as np
from stl import mesh as mesh2
from Helpers import *
from skimage import measure
from NetManager import INetManager


def _get_mesh(labels):
    vertices, faces, normals, values = measure.marching_cubes(labels, 0)
    my_mesh = mesh2.Mesh(np.zeros(faces.shape[0], dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = vertices[f[j], :]
    return my_mesh


def marching_cubes(net_manager: INetManager, sampling_resolution_3d):
    xyz = get_xyz_in_octant(None, sampling_resolution_3d)
    labels = net_manager.soft_predict(xyz).reshape(sampling_resolution_3d)
    return _get_mesh(labels)
