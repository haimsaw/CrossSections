import numpy as np


def csl_to_xyz(csl, save_path, n_points_per_edge):

    pts = []
    for plane in csl.planes:
        for cc in plane.connected_components:
            verts = plane.vertices[cc.vertices_indices]
            verts2 = np.concatenate((verts[1:], verts[0:1]))
            for e1, e2  in zip(verts, verts2):
                pts += [e1*d + e2*(1-d) for d in np.linspace(0, 1, n_points_per_edge)]

    pts = np.array(pts)
    file_name = f'{save_path}{csl.model_name}_{n_points_per_edge}_samples_per_edge.xyz'

    with open(file_name, 'w') as f:
        for pt in pts:
            f.write('{:.7f} {:.7f} {:.7f}\n'.format(*pt))
    print(file_name)

