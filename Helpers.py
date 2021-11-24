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


def get_mask_for_blending(xyzs, oct, oct_core, oct_direction):
    # return labels for blending in the x direction
    # xyzs are in octant+overlap
    # todo this assumes that octree depth is 1
    # todo refactor this

    core_end_x = oct_core[0][0]
    core_start_x = oct_core[1][0]

    margin_end_x = oct[0][0]
    margin_start_x = oct[1][0]

    non_blending_start_x = 2 * core_start_x - margin_start_x
    non_blending_end_x = 2 * core_end_x - margin_end_x

    core_end_y = oct_core[0][1]
    core_start_y = oct_core[1][1]

    margin_end_y = oct[0][1]
    margin_start_y = oct[1][1]

    non_blending_start_y = 2*core_start_y - margin_start_y
    non_blending_end_y = 2*core_end_y - margin_end_y

    core_end_z = oct_core[0][2]
    core_start_z = oct_core[1][2]

    margin_end_z = oct[0][2]
    margin_start_z = oct[1][2]

    non_blending_start_z = 2*core_start_z - margin_start_z
    non_blending_end_z = 2*core_end_z - margin_end_z

    if oct_direction[0] == '-':
        line_x = lambda xyz: xyz[0] * 1 / (non_blending_end_x - margin_end_x) + margin_end_x / (margin_end_x - non_blending_end_x)
    elif oct_direction[0] == '+':
        line_x = lambda xyz: xyz[0] * 1 / (non_blending_start_x - margin_start_x) + margin_start_x / (margin_start_x - non_blending_start_x)

    if oct_direction[1] == '-':
        line_y = lambda xyz: xyz[1] * 1 / (non_blending_end_y - margin_end_y) + margin_end_y / (margin_end_y - non_blending_end_y)
    elif oct_direction[1] == '+':
        line_y = lambda xyz: xyz[1] * 1 / (non_blending_start_y - margin_start_y) + margin_start_y / (margin_start_y - non_blending_start_y)

    if oct_direction[2] == '-':
        line_z = lambda xyz: xyz[2] * 1 / (non_blending_end_z - margin_end_z) + margin_end_z / (margin_end_z - non_blending_end_z)
    elif oct_direction[2] == '+':
        line_z = lambda xyz: xyz[2] * 1 / (non_blending_start_z - margin_start_z) + margin_start_z / (margin_start_z - non_blending_start_z)

    wights = np.array([min(1, line_x(xyz), line_y(xyz), line_z(xyz)) for xyz in xyzs])

    return wights
