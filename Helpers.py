import numpy as np


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)

# todo make class octent


def get_octs(top, btm, overlap_margin):
    # todo octets overlap only on x axis
    mid = (top + btm) / 2
    size = top - mid

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

    octanes_bottoms = np.array([top - size for top in octanes_tops])
    octs_core = np.stack((octanes_tops, octanes_bottoms), axis=1)

    overlap_size = size * overlap_margin
    octanes_tops += overlap_size
    octanes_bottoms -= overlap_size
    octs = np.stack((octanes_tops, octanes_bottoms), axis=1)

    return octs, octs_core


def get_xyzs_in_octant(octant, sampling_resolution_3d):
    x = np.linspace(-1, 1, sampling_resolution_3d[0])
    y = np.linspace(-1, 1, sampling_resolution_3d[1])
    z = np.linspace(-1, 1, sampling_resolution_3d[2])
    xyzs = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    if octant is not None:
        xyzs = xyzs[np.all(np.logical_and(xyzs <= octant[0], octant[1] <= xyzs), axis=1)]
    return xyzs


def is_in_octant_list(xyzs, top_bottom_octant):
    if top_bottom_octant is None:
        return np.full(True, xyzs.shape)
    return np.all(np.logical_and(xyzs <= top_bottom_octant[0], top_bottom_octant[1] <= xyzs), axis=1)


def is_in_octant(xyz, top_bottom_octant):
    # top_bottom_octant is a tuple (octant top, octant bottom), none for everywhere
    # todo remove this
    if top_bottom_octant is None:
        return True

    is_in_range_for_ax = (top > coordinate >= bottom
                          for coordinate, top, bottom in zip(xyz, *top_bottom_octant))
    return all(is_in_range_for_ax)


def get_mask_for_blending_old(xyzs, oct, oct_core, oct_direction):
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
    for i, direction in enumerate(oct_direction):
        lines.append(line_getter_pos(i) if direction == '+' else line_getter_neg(i))

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
