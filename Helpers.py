import numpy as np
from functools import wraps
from time import time


INSIDE_LABEL = 0.0
OUTSIDE_LABEL = 1.0


def get_top_bottom(points):
    top = np.amax(points, axis=0)
    bottom = np.amin(points, axis=0)
    return top, bottom


def add_margin(top, bottom, margin):
    return top + margin * (top - bottom),  bottom - margin * (top - bottom)


def get_xyzs_in_octant(oct, sampling_resolution_3d, endpoint=True):
    if oct is None:
        oct = [[1.0] * 3, [-1.0] * 3]
    x = np.linspace(oct[1][0], oct[0][0], sampling_resolution_3d[0], endpoint=endpoint)
    y = np.linspace(oct[1][1], oct[0][1], sampling_resolution_3d[1], endpoint=endpoint)
    z = np.linspace(oct[1][2], oct[0][2], sampling_resolution_3d[2], endpoint=endpoint)
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape((-1, 3))


def dot(a, b):
    return (a*b).sum(axis=1)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap