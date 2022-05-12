import numpy as np

from CSL import *
import pywavefront
from stl import mesh as mesh2


def make_csl_from_mesh(filename, save_path):
    verts, faces, scale = get_verts_faces(filename)
    model_name = filename.split('/')[-1].split('.')[0]

    if model_name == 'armadillo':
        plane_normals, ds = get_armadillo_planes(scale)
    elif model_name == "eight":
        plane_normals, ds = get_eight_planes(scale)
    elif model_name == "brain":
        plane_normals, ds = get_brain_planes(scale)

    else:
        plane_normals, ds = get_random_planes(scale)

    plane_normals = (plane_normals.T / np.linalg.norm(plane_normals.T, axis=0)).T
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]

    csl = CSL.from_mesh(model_name, plane_origins,  plane_normals, ds, verts, faces)

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
    scale = 1.1

    verts /= scale * np.max(np.absolute(verts))

    faces = scene.mesh_list[0].faces
    return verts, faces, 1/scale


def get_brain_planes(scale):
    n_slices = 150

    n_slices1 = n_slices // 3
    n_slices2 = n_slices1
    n_slices3 = n_slices - n_slices1 - n_slices2

    plane_normals = np.array([(1.0, 0, 0)] * n_slices1 + [(0, 1.0, 0)]*n_slices2 + [(0, 0.1, 1.0)]*n_slices3)
    ds = np.concatenate((np.linspace(-0.1, scale, n_slices1), np.linspace(-scale, scale, n_slices2), np.linspace(-0.7, 0.7, n_slices3)))
    return plane_normals, ds


def get_random_planes(scale):
    print('!!!!!! using random slices!!!!!!')
    n_slices = 50
    plane_normals = np.random.randn(n_slices, 3)

    ds = (np.random.random_sample(n_slices) * 2*scale - scale)
    return plane_normals, ds


def get_eight_planes(scale):
    n_slices = 30
    n_slices1 = n_slices - 1  # int(n_slices*0.85)
    n_slices2 = 1  # n_slices - n_slices1

    plane_normals = np.array([(0, 0, 1.0)] * n_slices1 + [(1, 0.0, 0.0)]*n_slices2)
    ds = np.concatenate((np.linspace(-scale, scale, n_slices1), [0]))

    return plane_normals, ds


def get_armadillo_planes(n_slices, scale):
    n_slices1 = int(n_slices*0.5)
    n_slices2 = n_slices - n_slices1

    plane_normals = np.array([(0, 1.0, 0)] * n_slices1 + [(0, 0, 1.0)]*n_slices2)
    ds = np.concatenate((np.linspace(-0.9, 0.7, n_slices1), np.linspace(-0.7, 0.7, n_slices2)))

    return plane_normals, ds

