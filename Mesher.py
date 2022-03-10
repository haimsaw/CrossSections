import math

import numpy as np
from stl import mesh as mesh2
from Helpers import *
from skimage import measure
from NetManager import INetManager
from mcdc.dual_contour_3d import dual_contour_3d
from sklearn.preprocessing import normalize


def _get_mesh(labels, level, spacing):
    vertices, faces, normals, values = measure.marching_cubes(labels, level=level, spacing=spacing)
    vertices = vertices - 1  # center mesh
    my_mesh = mesh2.Mesh(np.zeros(faces.shape[0], dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = vertices[f[j], :]
    return my_mesh


@timing
def marching_cubes(net_manager: INetManager, sampling_resolution_3d, use_sigmoid):
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)

    labels = net_manager.soft_predict(xyzs, use_sigmoid=use_sigmoid)
    print(f'max={max(labels)} min={min(labels)}')

    if use_sigmoid:
        # set level is at 0
        labels = labels * 2 - 1

    # use spacing to match original shape boundaries
    return _get_mesh(labels.reshape(sampling_resolution_3d), level=0, spacing=[2/res for res in sampling_resolution_3d])


@timing
def dual_contouring(net_manager: INetManager, sampling_resolution_3d, use_grads, use_sigmoid):
    sampling_resolution_3d = np.array(sampling_resolution_3d)

    # since in dc our vertices are inside the grid cells we need to have res+1 grid points
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d+1, endpoint=True)

    labels = net_manager.soft_predict(xyzs, use_sigmoid=use_sigmoid).reshape(sampling_resolution_3d+1)

    # set level is at 0 so normalize labels to be in [-1, 1]
    if use_sigmoid:
        labels = labels * 2 - 1
    # print(f'labels max={labels.max()} min={labels.min()}')

    # dual_contour_3d uses grid points as coordinates
    # so i j k are the indices for the label (and not the actual point)
    def f(i, j, k):
        '''d0 = np.array([i, j, k]) - center0
        d1 = np.array([i, j, k]) - center1
        return (np.dot(d0, d0) - radius0 ** 2) - (np.dot(d1, d1) - radius1) ** 2'''

        return labels[i][j][k]

    def get_f_normal(ijks_for_normal):

        if use_grads is True and len(ijks_for_normal) > 0:
            # translate from ijk (index) coordinate system to xyz
            # where xyz = np.linspace(-1, 1, sampling_resolution_3d[i]+1, endpoint=True)
            ijks_for_normal = np.array(ijks_for_normal)
            xyzs_for_normal = 2 * ijks_for_normal / (sampling_resolution_3d + 1) - 1

            normals = -1 * net_manager.grad_wrt_input(xyzs_for_normal, use_sigmoid=use_sigmoid)
            # normals = normalize(normals, norm="l2")  # todo haim?

            # print(f'use_grads={use_grads} avg={np.abs(normals).mean(axis=0)}')

            ijks_to_grad = dict(zip(map(tuple, ijks_for_normal), normals))
            return lambda i, j, k: ijks_to_grad[(i, j, k)]

        else:
            return lambda i, j, k: np.array([0.0, 0.0, 0.0])

    return dual_contour_3d(f, get_f_normal, *sampling_resolution_3d)


