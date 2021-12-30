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
    grads = net_manager.grad_wrt_input(xyzs)

    # set level is at 0
    labels = labels * 2 - 1

    xyzs = map(tuple, xyzs)

    center = np.array([25.0, 25.0, 25.0])
    radius = 25

    '''
    dual_contour_3d uses grid points as coordinates
    so i j k are the indices for the the label (and not the actual point)  
    '''
    def f(i, j, k):
        return labels[i][j][k]

        d = np.array([i, j, k]) - center
        return np.dot(d, d) - radius ** 2

    def df(x, y, z):
        d = np.array([x, y, z]) - center
        return d / math.sqrt(np.dot(d, d))

    def get_f_normal(ijks_for_normal):
        xyzs_for_normal = np.array(ijks_for_normal) / sampling_resolution_3d  # todo haim - is this correct?
        ijks_to_grad = dict(zip(map(tuple, ijks_for_normal), net_manager.grad_wrt_input(xyzs_for_normal)))
        return lambda i, j, k: ijks_to_grad[(i, j, k)]

    return dual_contour_3d(f, get_f_normal,
                           sampling_resolution_3d[0]-1,
                           sampling_resolution_3d[1]-1,
                           sampling_resolution_3d[2]-1)
