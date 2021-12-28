import numpy as np


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)


def get_xyzs_in_octant(oct, sampling_resolution_3d):
    if oct is None:
        oct = [[1.0] * 3, [-1.0] * 3]
    x = np.linspace(oct[1][0], oct[0][0], sampling_resolution_3d[0])
    y = np.linspace(oct[1][1], oct[0][1], sampling_resolution_3d[1])
    z = np.linspace(oct[1][2], oct[0][2], sampling_resolution_3d[2])
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape((-1, 3))

