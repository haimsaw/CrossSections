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


def get_mask_for_blending_old(xyzs, oct, oct_core):
    # return labels for blending in the x direction
    # xyzs are in octant+overlap
    # todo this assumes that octree depth is 1
    # todo refactor this

    core_end = oct_core[0]
    core_start = oct_core[1]
    margin_end = oct[0]
    margin_start = oct[1]

    non_blending_start = 2 * core_start - margin_start
    non_blending_end = 2 * core_end - margin_end

    line_getter_pos = lambda i: lambda xyz: xyz[i] * 1 / (non_blending_start[i] - margin_start[i]) + margin_start[i] / (margin_start[i] - non_blending_start[i])
    line_getter_neg = lambda i: lambda xyz: xyz[i] * 1 / (non_blending_end[i] - margin_end[i]) + margin_end[i] / (margin_end[i] - non_blending_end[i])

    lines = []
    for i in range(3):
        lines.append( line_getter_neg(i))
        lines.append(line_getter_pos(i))

    wights = np.array([min(1, *[l(xyz) for l in lines]) for xyz in xyzs])

    return wights


def get_mask_for_blending(xyzs, oct, oct_core, oct_direction):
    # return labels for blending in the x direction
    # xyzs are in octant+overlap
    # todo this assumes that octree depth is 1
    # todo refactor this

    # 3 1d interpolation (1 chose 3)
    # 3 2d interpolation (2 chose 3)
    # 1 3d interpolation (3 chose 2)

    core_start = oct_core[1]
    core_end = oct_core[0]

    margin_start = oct[1]
    margin_end = oct[0]

    non_blending_start = 2 * core_start - margin_start
    non_blending_end = 2 * core_end - margin_end

    wights = np.full(xyzs.shape, 1.0)

    # 3d interpolation
    points = []

    return wights
