from math import radians
from random import random
from Helpers import *
import numpy as np
import matplotlib.pyplot as plt
import inspect

from Resterizer import rasterizer_factory
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from NetManager import INetManager

# region 3d


def _get_3d_ax():
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    fig.suptitle(inspect.stack()[1][3])
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax


class Renderer3D:
    def __init__(self):
        self.ax = _get_3d_ax()

    def add_scene(self, csl):

        # colors = [[random(), random(), random()] for _ in range(csl.n_labels + 1)]

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    vertices = plane.vertices[connected_component.vertices_indices_in_component]
                    vertices[-1] = vertices[0]
                    alpha = 1 if connected_component.is_hole else 0.5
                    # ax.plot_trisurf(*vertices.T, color='green', alpha=alpha)
                    self.ax.plot(*vertices.T, color='green')
                    # ax.plot_surface(*vertices.T, color='green')

    def add_dataset(self, dataset):

        xyzs = np.array([xyz.detach().numpy() for xyz, label in dataset if label == 1])

        self.ax.scatter(*xyzs.T, color="blue")

    def add_rasterized_scene(self, csl, sampling_resolution_2d, sampling_margin, show_empty_planes=True, show_outside_shape=False, alpha=0.1):

        for plane in csl.planes:
            cells = rasterizer_factory(plane).get_rasterazation_cells(sampling_resolution_2d, sampling_margin)
            mask = np.array([cell.label for cell in cells])
            xyzs = np.array([cell.xyz for cell in cells])

            if not plane.is_empty:
                self.ax.scatter(*xyzs[mask].T, color="blue", alpha=alpha)
                if show_outside_shape:
                    self.ax.scatter(*xyzs[np.logical_not(mask)].T, color="gray", alpha=alpha)
            elif show_empty_planes:
                self.ax.scatter(*xyzs.T, color="purple", alpha=alpha/2)

    def add_model_hard_prediction(self, network_manager: INetManager, sampling_resolution_3d, alpha=0.05, octant=None):
        xyzs = get_xyzs_in_octant(octant, sampling_resolution_3d)
        xyzs, labels = network_manager.hard_predict(xyzs)

        self.ax.scatter(*xyzs[labels].T, alpha=alpha, color='blue')

    def add_model_soft_prediction(self, network_manager: INetManager, sampling_resolution_3d, alpha=1.0):
        # todo not working

        x = np.linspace(-1, 1, sampling_resolution_3d[0])
        y = np.linspace(-1, 1, sampling_resolution_3d[1])
        z = np.linspace(-1, 1, sampling_resolution_3d[2])

        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
        _, labels = network_manager.soft_predict(xyz)

        self.ax.scatter(*xyz.T, c=labels, cmap="Blues", alpha=alpha)

    def add_mesh(self, my_mesh):

        collection = Poly3DCollection(my_mesh.vectors)
        collection.set_edgecolor('k')

        self.ax.add_collection3d(collection)

        scale = my_mesh.points.flatten()
        self.ax.auto_scale_xyz(scale, scale, scale)

    def add_model_errors(self, network_manager: INetManager):
        errored_xyz, errored_labels = network_manager.get_train_errors()
        self.ax.scatter(*errored_xyz[errored_labels == 1].T, color="blue")
        self.ax.scatter(*errored_xyz[errored_labels == 0].T, color="red")

    def show(self):
        plt.show()

# endregion


# region 2d


# todo make this a class like the Renderer3D
def draw_rasterized_plane(plane, resolution=(256, 256), margin=0.2):
    plt.imshow(rasterizer_factory(plane).get_rasterazation_cells(resolution, margin)[0].reshape(resolution), cmap='cool',
               origin='lower')
    plt.suptitle("draw_rasterized_plane")
    plt.show()


def draw_plane_verts(plane):
    verts, _ = plane.pca_projection
    for component in plane.connected_components:
        plt.scatter(*verts[component.vertices_indices_in_component].T, color='orange' if component.is_hole else 'black')
    plt.scatter([0], [0], color='red')
    plt.suptitle("draw_plane_verts")
    plt.show()


def draw_plane(plane):
    verts, _ = plane.pca_projection
    for component in plane.connected_components:
        plt.plot(*verts[component.vertices_indices_in_component].T, color='orange' if component.is_hole else 'black')
    plt.suptitle("draw_plane")
    plt.show()

# endregion
