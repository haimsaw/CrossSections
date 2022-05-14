import numpy as np


def csl_to_ply(csl, save_path, n_points_per_edge=3):
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
    normals = np.array(normals)

    header = f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\n' \
             f'property float x\nproperty float y\nproperty float z\n' \
             f'property float nx\nproperty float ny\nproperty float nz\n' \
             f'element face 0\nproperty list uchar int vertex_index\nend_header'

    file_name = f'{save_path}{csl.model_name}.ply'

    with open(file_name, 'w') as f:
        f.write(header)
        for pt in pts:
            f.write('{:.10f} {:.10f} {:.10f}\n'.format(*pt))

    print(f'{file_name}')
