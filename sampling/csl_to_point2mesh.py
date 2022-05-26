import numpy as np
import trimesh


def csl_to_point2mesh(csl, save_path, mesh_path, n_points_per_edge=3):
    pts = []
    for plane in csl.planes:
        for cc in plane.connected_components:
            verts = plane.vertices[cc.vertices_indices]
            verts2 = np.concatenate((verts[1:], verts[0:1]))
            for e1, e2 in zip(verts, verts2):
                pts += [e1 * d + e2 * (1 - d) for d in np.linspace(0, 1, n_points_per_edge)]

    pts = np.array(pts)

    scaled_mesh = trimesh.load_mesh(mesh_path)

    closest_points, distances, triangle_ids = scaled_mesh.nearest.on_surface(pts)
    normals = scaled_mesh.face_normals[triangle_ids]

    header = f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\n' \
             f'property float x\nproperty float y\nproperty float z\n' \
             f'property float nx\nproperty float ny\nproperty float nz\n' \
             f'element face 0\nproperty list uchar int vertex_index\nend_header\n'

    file_name = f'{save_path}{csl.model_name}.ply'

    with open(file_name, 'w') as f:
        f.write(header)
        for pt, n in zip(pts, normals):
            f.write('{:.10f} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f}\n'.format(*pt, *n))

    print(f'{file_name}')
