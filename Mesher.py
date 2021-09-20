import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid


def marching_cubes(network_manager, sampling_resolution):

    x = np.linspace(-1, 1, sampling_resolution[0])
    y = np.linspace(-1, 1, sampling_resolution[1])
    z = np.linspace(-1, 1, sampling_resolution[2])

    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    labels = network_manager.predict(xyz).reshape(sampling_resolution)

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(labels, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(*verts.T)
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 60)
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    plt.show()