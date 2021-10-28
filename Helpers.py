import numpy as np


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)


def get_octets(top, bottom):
    # todo octets with overlap
    mid = (top + bottom) / 2
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
    return np.stack((octanes_tops, octanes_bottoms), axis=1)


def get_xyz_in_octant(octant, sampling_resolution_3d):
    x = np.linspace(-1, 1, sampling_resolution_3d[0])
    y = np.linspace(-1, 1, sampling_resolution_3d[1])
    z = np.linspace(-1, 1, sampling_resolution_3d[2])
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    if octant is not None:
        xyz = xyz[np.all(np.logical_and(xyz <= octant[0], octant[1] <= xyz), axis=1)]
    return xyz
