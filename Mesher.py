import math

import numpy as np
from stl import mesh as mesh2
from Helpers import *
from skimage import measure
from NetManager import INetManager
from mcdc.dual_contour_3d import dual_contour_3d


def _get_mesh(labels, level, spacing):
    vertices, faces, normals, values = measure.marching_cubes(labels, level=level, spacing=spacing)
    vertices = vertices - 1  # center mesh
    my_mesh = mesh2.Mesh(np.zeros(faces.shape[0], dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = vertices[f[j], :]
    return my_mesh


def marching_cubes(net_manager: INetManager, sampling_resolution_3d):
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)

    labels = net_manager.soft_predict(xyzs)
    # argsorted_xyzs = np.lexsort((xyzs.T[2], xyzs.T[1], xyzs.T[0]))

    # use spacing to match original shape boundaries
    #return _get_mesh(labels[argsorted_xyzs].reshape(sampling_resolution_3d), level=0.5, spacing=[2/res for res in sampling_resolution_3d])
    return _get_mesh(labels.reshape(sampling_resolution_3d), level=0.5, spacing=[2/res for res in sampling_resolution_3d])


def dual_contouring(net_manager: INetManager, sampling_resolution_3d):
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)
    labels = net_manager.soft_predict(xyzs).reshape(sampling_resolution_3d)

    # set level is at 0
    labels = labels * 2 - 1

    xyzs = map(tuple, xyzs)

    center = np.array([0.0, 0.0, 0.0])
    radius = 15.0

    '''
    dual_contour_3d uses grid points as coordinates
    so i j k are the indexs for the the label (and not the actual point)  
    '''
    def f(i, j, k):
        return labels[i][j][k]

    def df(x, y, z):
        x = np.array([x, y, z])
        d = x - center
        return d / math.sqrt(np.dot(d, d))

    return dual_contour_3d(f, df, sampling_resolution_3d,
                           0, sampling_resolution_3d[0]-1,
                           0, sampling_resolution_3d[1]-1,
                           0, sampling_resolution_3d[2]-1)
