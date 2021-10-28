from math import radians
from random import random
from Helpers import *
import glfw
import numpy as np
from OpenGL.GL import *
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import inspect

from Resterizer import rasterizer_factory
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# region OpenGL

class RendererOpenGL:
    def __init__(self, csl, box):
        self.csl = csl

        self.cursor_origin = np.zeros(2, dtype='float64')

        self.rotation = Quaternion(axis=(1, 0, 0), radians=0)
        self.translation = np.zeros(2, dtype='float64')
        self.zoom = 0.5

        self.colors = [[random(), random(), random()] for _ in range(self.csl.n_labels + 1)]
        self.box = box

        glfw.init()
        self.window = glfw.create_window(1600, 1200, "Cross Sections", None, None)
        glfw.set_window_pos(self.window, 400, 100)
        glfw.make_context_current(self.window)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glMatrixMode(GL_MODELVIEW)

        glfw.set_scroll_callback(self.window, self.__get_on_scroll())
        glClearColor(0, 0.1, 0.1, 1)

    def __get_on_scroll(self):
        def on_scroll(window, dx, dy):
            self.zoom += dy * 0.1

        return on_scroll

    def __draw_scene(self):
        for plane in self.csl.planes:
            for connected_component in plane.connected_components:
                vertices = plane.vertices[connected_component.vertices_indices_in_component]
                self.__draw_vertices(vertices, connected_component.label, GL_LINE_LOOP)
        self.__draw_vertices(self.box, self.csl.n_labels, GL_LINE_LOOP)

    def __draw_vertices(self, vertices: np.array, label, mode):
        v = np.array(vertices.flatten(), dtype=np.float32)
        glVertexPointer(3, GL_FLOAT, 0, v)

        color = self.colors[label] * len(vertices)
        color = np.array(color, dtype=np.float32)
        glColorPointer(3, GL_FLOAT, 0, color)

        glDrawArrays(mode, 0, len(vertices))

    def __handle_mouse_events(self):
        cur_cursor_pos = np.array(glfw.get_cursor_pos(self.window), dtype='float64')

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            d_theta = (self.cursor_origin[0] - cur_cursor_pos[0]) / 10
            d_rho = (self.cursor_origin[1] - cur_cursor_pos[1]) / 10

            rotation_theta = Quaternion(axis=(0, 1, 0), angle=radians(d_theta / 2))
            rotation_rho = Quaternion(axis=(1, 0, 0), angle=radians(d_rho / 2))

            rotation = rotation_theta * rotation_rho
            self.rotation = (rotation * self.rotation).normalised

        elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            d_translation = (self.cursor_origin - cur_cursor_pos) / 200
            d_translation[0] *= -1
            self.translation += d_translation

        self.cursor_origin = cur_cursor_pos

    def __do_transformations(self):
        glLoadIdentity()
        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslate(*self.translation, 0)
        glRotatef(self.rotation.degrees, *self.rotation.axis)

    def event_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT)

            self.__handle_mouse_events()
            self.__do_transformations()

            self.__draw_scene()

            glfw.swap_buffers(self.window)

        glfw.terminate()

# endregion


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

    def add_model_hard_prediction(self, network_manager, sampling_resolution_3d, alpha=0.05, octant=None):
        xyz = get_xyz_in_octant(octant, sampling_resolution_3d)
        labels = network_manager.hard_predict(xyz)

        self.ax.scatter(*xyz[labels].T, alpha=alpha, color='blue')

    def add_model_soft_prediction(self, network_manager, sampling_resolution_3d, alpha=1.0):
        # todo not working

        x = np.linspace(-1, 1, sampling_resolution_3d[0])
        y = np.linspace(-1, 1, sampling_resolution_3d[1])
        z = np.linspace(-1, 1, sampling_resolution_3d[2])

        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
        labels = network_manager.hard_predict(xyz)

        self.ax.scatter(*xyz.T, c=labels, cmap="Blues", alpha=alpha)

    def add_mesh(self, my_mesh):

        collection = Poly3DCollection(my_mesh.vectors)
        collection.set_edgecolor('k')

        self.ax.add_collection3d(collection)

        scale = my_mesh.points.flatten()
        self.ax.auto_scale_xyz(scale, scale, scale)

        plt.show()

    def add_model_errors(self, network_manager):
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
