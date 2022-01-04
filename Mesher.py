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


@timing
def marching_cubes(net_manager: INetManager, sampling_resolution_3d):
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)

    # set level is at 0
    labels = net_manager.soft_predict(xyzs) * 2 - 1

    # use spacing to match original shape boundaries
    return _get_mesh(labels.reshape(sampling_resolution_3d), level=0, spacing=[2/res for res in sampling_resolution_3d])


@timing
def dual_contouring(net_manager: INetManager, sampling_resolution_3d, trans):
    xyzs = get_xyzs_in_octant(None, sampling_resolution_3d, endpoint=False)
    labels = net_manager.soft_predict(xyzs).reshape(sampling_resolution_3d)

    # set level is at 0
    labels = labels * 2 - 1

    # dual_contour_3d uses grid points as coordinates
    # so i j k are the indices for the label (and not the actual point)
    def f(i, j, k):
        return labels[i][j][k]

    def get_f_normal(ijks_for_normal):

        if trans == 0:
            xyzs_for_normal = np.array(ijks_for_normal) / sampling_resolution_3d  # todo haim - is this correct?

        elif trans == 1:
            # translate from ijk (index) corodinate system to xyz
            # where xyz = np.linspace(-1, 1, sampling_resolution_3d[i]-1, endpoint=False)

            radius = np.array(sampling_resolution_3d) / 2
            xyzs_for_normal = np.array(ijks_for_normal) / radius - 1

        elif trans == 2:
            radius = np.array(sampling_resolution_3d)
            xyzs_for_normal = np.array(ijks_for_normal) / radius - 1

        elif trans == 3:
            return lambda i, j, k: np.array([0.0, 0.0, 0.0])

        ijks_to_grad = dict(zip(map(tuple, ijks_for_normal), net_manager.grad_wrt_input(xyzs_for_normal)))
        return lambda i, j, k: ijks_to_grad[(i, j, k)]

    return dual_contour_3d(f, get_f_normal,
                           sampling_resolution_3d[0] - 1,  # todo haim -1 +1?
                           sampling_resolution_3d[1] - 1,
                           sampling_resolution_3d[2] - 1)


