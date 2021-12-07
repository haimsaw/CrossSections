import numpy as np


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)


def get_xyzs_in_octant(octant, sampling_resolution_3d):
    x = np.linspace(-1, 1, sampling_resolution_3d[0])
    y = np.linspace(-1, 1, sampling_resolution_3d[1])
    z = np.linspace(-1, 1, sampling_resolution_3d[2])
    xyzs = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    if octant is not None:
        xyzs = xyzs[np.all(np.logical_and(xyzs <= octant[0], octant[1] <= xyzs), axis=1)]
    return xyzs


