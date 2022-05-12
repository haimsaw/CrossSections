import numpy as np
from matplotlib import pyplot as plt

from hp import get_csl, HP

def csl_to_xyz():
    save_path = "xyz-files/"
    hp = HP()
    n_points_per_edge = 5

    csl = get_csl(hp.bounding_planes_margin, save_path)
    pts = []
    for plane in csl.planes:
        for cc in plane.connected_components:
            verts = plane.vertices[cc.vertices_indices]
            verts2 = np.concatenate((verts[1:], verts[0:1]))
            for e1, e2  in zip(verts, verts2):
                pts += [e1*d + e2*(1-d) for d in np.linspace(0, 1, n_points_per_edge)]

    pts = np.array(pts)
    file_name = f'{save_path}{csl.model_name}.xyz'

    with open(file_name, 'w') as f:
        for pt in pts:
            f.write('{:.10f} {:.10f} {:.10f}\n'.format(*pt))
    print(file_name)


if __name__ == "__main__":
    csl_to_xyz()