from multiprocessing import Pool, cpu_count

import numpy as np

from sampling.CSL import *
import pywavefront
from stl import mesh as mesh2
from  Renderer import *
from meshcut import cross_section

from sampling.csl_to_point2mesh import csl_to_point2mesh
from sampling.csl_to_xyz import csl_to_xyz


class GetCcs:
    def __init__(self, verts, faces):
        self.verts=verts
        self.faces= faces

    def __call__(self, orig_normal):
        o, n = orig_normal[0], orig_normal[1]
        return cross_section(self.verts, self.faces, plane_orig=o, plane_normal=n)


def make_csl_from_mesh(filename, save_path):
    verts, faces, normals, scale = get_verts_faces(filename)
    model_name = filename.split('/')[-1].split('.')[0]

    top, bottom = get_top_bottom(verts)
    top = top * 0.97
    bottom = bottom * 0.97

    if 'armadillo' in model_name:
        plane_normals, ds = get_armadillo_planes(scale, top, bottom)
    elif "eight_15" in model_name:
        plane_normals, ds = get_eight_15_planes(scale, top, bottom)
    elif "eight_20" in model_name:
        plane_normals, ds = get_eight_20_planes(scale, top, bottom)
    elif "brain" in model_name:
        plane_normals, ds = get_brain_planes(scale, top, bottom)
    elif 'lamp004_fixed' in model_name:
        plane_normals, ds = get_lamp_planes(scale, top, bottom)
    elif 'SeaShell_4_fixed' in model_name:
        plane_normals, ds = get_shell_planes(scale, top, bottom)
    else:
        plane_normals, ds = get_random_planes(scale, top, bottom)

    plane_normals = (plane_normals.T / np.linalg.norm(plane_normals.T, axis=0)).T
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]

    ccr = GetCcs(verts, faces)
    ccs_per_plane = list(map(ccr, zip(plane_origins, plane_normals)))

    csl = CSL.from_mesh(model_name, plane_origins,  plane_normals, ds, ccs_per_plane)

    with open(f'{save_path}/{model_name}_from_mesh.csl', 'w') as f:
        f.write(repr(csl))

    my_mesh = mesh2.Mesh(np.zeros(len(faces), dtype=mesh2.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j], :]

    mesh_path = f'{save_path}/{model_name}_scaled.stl'
    my_mesh.save(mesh_path)
    print(f'csl={csl.model_name} slices={len([p for p in csl.planes if not p.is_empty])}, n edges={len(csl)}')

    # RendererPoly.init()
    # RendererPoly.add_mesh(verts,faces)
    # RendererPoly.add_scene(csl)
    # RendererPoly.show()

    csl_to_point2mesh(csl, './data/for_pt2mesh/', mesh_path,)
    # csl_to_xyz(csl, './data/for_vipss/', 1)
    return csl


def get_verts_faces(filename):
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    assert len(scene.mesh_list) == 1

    verts = np.array(scene.vertices)
    verts -= np.mean(verts, axis=0)
    scale = 1.1

    verts /= scale * np.max(np.absolute(verts))

    faces = scene.mesh_list[0].faces
    normals = np.array(scene.parser.normals)

    assert normals.shape == verts.shape
    return verts, faces, normals, 1/scale


def get_brain_planes(scale, top, bottom):
    n_slices = 150

    n_slices1 = n_slices // 3
    n_slices2 = n_slices1
    n_slices3 = n_slices - n_slices1 - n_slices2

    plane_normals = np.array([(1.0, 0, 0)] * n_slices1 +
                             [(0, 1.0, 0)] * n_slices2 +
                             [(0, 0, 1.0)] * n_slices3)

    ds = -1 * np.concatenate((np.linspace(bottom[0], top[0], n_slices1),
                              np.linspace(bottom[1], top[1], n_slices2),
                              np.linspace(bottom[2], top[2], n_slices3)))
    return plane_normals, ds


def get_random_planes(scale, top, bottom):
    print('!!!!!! using random slices!!!!!!')
    n_slices = 50
    plane_normals = np.random.randn(n_slices, 3)

    ds = -1 * (np.random.random_sample(n_slices) * 2* scale - scale)
    return plane_normals, ds


def get_lamp_planes(scale, top, bottom):
    n_slices = 150
    plane_normals = np.random.randn(n_slices, 3)

    ds = -1 * np.random.normal(size=n_slices)
    return plane_normals, ds


def get_shell_planes(scale, top, bottom):
    n_slices = 50
    plane_normals = np.random.randn(n_slices, 3)

    ds = -1 * (np.random.random_sample(n_slices) * 2 * scale - scale)
    return plane_normals, ds


def get_eight_15_planes(scale, top, bottom):
    n_slices = 15

    n_slices_x = 0  # n_slices - n_slices1
    n_slices_z = n_slices - n_slices_x  # int(n_slices*0.85)

    plane_normals = np.array([(0, 0, 1.0)] * n_slices_z +
                             [(1.0, 0, 0.)]*n_slices_x)
    ds = -1 * np.concatenate((np.linspace(bottom[2], top[2], n_slices_z),
                              np.linspace(bottom[0], top[0], n_slices_x)))

    return plane_normals, ds


def get_eight_20_planes(scale, top, bottom):
    n_slices = 20

    n_slices_x = 5  # n_slices - n_slices1
    n_slices_z = n_slices - n_slices_x  # int(n_slices*0.85)

    plane_normals = np.array([(0, 0, 1.0)] * n_slices_z +
                             [(1, 0.0, 0.0)]*n_slices_x)
    ds = -1 * np.concatenate((np.linspace(bottom[2], top[2], n_slices_z),
                              np.linspace(bottom[0], top[0], n_slices_x)))

    return plane_normals, ds


def get_armadillo_planes(scale, top, bottom):
    n_slices = 25

    n_slices_y = 20
    n_slices2 = 3
    n_slices3 = 3

    plane_normals = np.array([(0, 1.0, 0)] * n_slices_y +
                             [(0,  1.3,  -1.0)] * n_slices2 +
                             # [(0, 1.0, 1.5)] * n_slices3)
                             [(-0.024,  0.050,  0.200)]*n_slices3) # this samples his fingers

    ds = -1 * np.concatenate((np.linspace(bottom[1], top[1], n_slices_y),
                              np.linspace(-0.5, 0.5, n_slices2),
                              # np.linspace(-0.62, 0.62, n_slices3)))
                              [0.511, 0.566, 0.622]))  # this samples his fingers

    return plane_normals, ds

