import numpy as np
from stl import mesh

from skimage import measure


def marching_cubes(network_manager, sampling_resolution):
    x = np.linspace(-1, 1, sampling_resolution[0])
    y = np.linspace(-1, 1, sampling_resolution[1])
    z = np.linspace(-1, 1, sampling_resolution[2])

    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    labels = network_manager.predict(xyz).reshape(sampling_resolution)

    vertices, faces, normals, values = measure.marching_cubes_lewiner(labels, 0)

    my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = vertices[f[j], :]

    return my_mesh
