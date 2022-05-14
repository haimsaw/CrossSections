import numpy as np
from plyfile import PlyData, PlyElement


def csl_to_xyz(csl, save_path, n_points_per_edge=1):
    pts = []
    normals = []
    for plane in csl.planes:
        for cc in plane.connected_components:
            verts = plane.vertices[cc.vertices_indices]
            verts2 = np.concatenate((verts[1:], verts[0:1]))
            for e1, e2 in zip(verts, verts2):
                pts += [e1 * d + e2 * (1 - d) for d in np.linspace(0, 1, n_points_per_edge)]
                pt_normal = np.cross(e2 - e1, plane.normal)
                pt_normal /= np.linalg.norm(pt_normal)

                normals += [pt_normal]

    pts = np.array(pts)

    vertex = np.array([(0, 0, 0),
                       (0, 1, 1),
                       (1, 0, 1),
                       (1, 1, 0)],
                      dtype=[('x', 'f4'), ('y', 'f4'),
                             ('z', 'f4')])

    pts = pts[np.random.permutation(len(pts))[:4000]]  # vipss can handle ~6k points

    file_name = f'{save_path}{csl.model_name}.xyz'

    with open(file_name, 'w') as f:
        for pt in pts:
            f.write('{:.10f} {:.10f} {:.10f}\n'.format(*pt))

    print(f'{file_name} {[len(set(pts[:, i])) for i in [0, 1, 2]]}')
