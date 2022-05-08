import numpy as np

from CSL import *
import pywavefront
from stl import mesh as mesh2


def make_csl_from_mesh(filename, save_path):
    n_slices = 20
    plane_origins = [(0, 0, d) for d in np.linspace(-1, 1, n_slices)]
    plane_normals = [(0, 0, 1.0)] * n_slices

    # todo make sure the normals are normlized
    verts, faces = get_verts_faces(filename)
    model_name = filename.split('/')[-1].split('.')[0]

    csl = CSL.from_mesh(model_name, plane_origins,  plane_normals, verts, faces)

    with open(f'{save_path}/{model_name}_generated.csl', 'w') as f:
        f.write(repr(csl))

    my_mesh = mesh2.Mesh(np.zeros(len(faces), dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j], :]

    mesh_path = f'{save_path}/original_mesh.stl'
    my_mesh.save(mesh_path)
    return csl


def get_verts_faces(filename):
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    assert len(scene.mesh_list) == 1
    verts = np.array(scene.vertices)
    verts -= np.mean(verts, axis=0)
    verts /= np.max(np.absolute(verts))
    faces = scene.mesh_list[0].faces
    return verts, faces


if __name__ == '__main__':
    make_csl_from_mesh()
