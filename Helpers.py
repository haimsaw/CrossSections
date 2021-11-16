import numpy as np
import torch


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)

# todo make class octent


def get_octets(top, btm, overlap_margin):
    # todo octets overlap only on x axis
    mid = (top + btm) / 2
    size = top - mid
    overlap = size * overlap_margin

    # in comment the numbers in https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Cube_with_balanced_ternary_labels.svg/800px-Cube_with_balanced_ternary_labels.svg.png
    # green for top, red for bottom
    octanes_tops = np.array(
            [[top[0], top[1], top[2]],  # 13
             [mid[0], top[1], top[2]],  # 12
             [mid[0], mid[1], top[2]],  # 9
             [top[0], mid[1], top[2]],  # 10

             [top[0], top[1], mid[2]],  # 4
             [mid[0], top[1], mid[2]],  # 3
             [mid[0], mid[1], mid[2]],  # 0
             [top[0], mid[1], mid[2]],  # 1
             ])

    octanes_bottoms = np.array(
            [[mid[0], mid[1], mid[2]],  # 0
             [btm[0], mid[1], mid[2]],  # 1
             [btm[0], btm[1], mid[2]],  # 4
             [mid[0], btm[1], mid[2]],  # 3

             [mid[0], mid[1], btm[2]],  # 9
             [btm[0], mid[1], btm[2]],  # 10
             [btm[0], btm[1], btm[2]],  # 13
             [mid[0], btm[1], btm[2]],  # 12
             ])
    # octanes_bottoms = np.array([top - size for top in octanes_tops])
    return np.stack((octanes_tops, octanes_bottoms), axis=1)


def get_xyzs_in_octant(octant, sampling_resolution_3d):
    x = np.linspace(-1, 1, sampling_resolution_3d[0])
    y = np.linspace(-1, 1, sampling_resolution_3d[1])
    z = np.linspace(-1, 1, sampling_resolution_3d[2])
    xyzs = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    if octant is not None:
        xyzs = xyzs[np.all(np.logical_and(xyzs < octant[0], octant[1] <= xyzs), axis=1)]
    return xyzs


def is_in_octant_list(xyzs, top_bottom_octant):
    if top_bottom_octant is None:
        return np.full(True, xyzs.shape)
    return np.all(np.logical_and(xyzs < top_bottom_octant[0], top_bottom_octant[1] <= xyzs), axis=1)


def is_in_octant(xyz, top_bottom_octant):
    # top_bottom_octant is a tuple (octant top, octant bottom), none for everywhere
    # todo remove this
    if top_bottom_octant is None:
        return True

    is_in_range_for_ax = (top > coordinate >= bottom
                          for coordinate, top, bottom in zip(xyz, *top_bottom_octant))
    return all(is_in_range_for_ax)


def get_mask_for_merging(xyzs, octant):
    # return labels for belnding in the x direction (from - to +)
    # xyzs are in octant+overlap

    top_x = octant[0][0]
    overlap_end_x = get_top_bottom(xyzs)[0][0]
    line = lambda xyz: xyz[0] * 1 / (top_x - overlap_end_x) + overlap_end_x / (overlap_end_x - top_x)
    wights = np.array([min(1, line(xyz)) for xyz in xyzs])

    return wights






































