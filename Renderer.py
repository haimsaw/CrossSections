from math import radians
from random import random

import glfw
import numpy as np
from OpenGL.GL import *
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

from Resterizer import rasterizer_factory


class Renderer:
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
                vertices = plane.vertices[connected_component.vertices_indeces_in_component]
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
            d_translation = (self.cursor_origin - cur_cursor_pos)/200
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


def draw_scene(csl, box):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    colors = [[random(), random(), random()] for _ in range(csl.n_labels + 1)]

    for plane in csl.planes:
        for connected_component in plane.connected_components:
            vertices = plane.vertices[connected_component.vertices_indeces_in_component]
            alpha = 1 if connected_component.is_hole else 0.5
            ax.plot_trisurf(*vertices.T, color=colors[connected_component.label], alpha=alpha)
    # todo show box
    # ax.plot_trisurf(*box.T, color=colors[-1])
    fig.suptitle("draw_scene")
    plt.show()
    # self.__draw_vertices(self.box, self.csl.n_labels, GL_LINE_LOOP)


def draw_rasterized_plane(plane, resolution=(256, 256), margin=0.2):
    plt.imshow(rasterizer_factory(plane).get_rasterazation(resolution, margin)[0].reshape(resolution), cmap='cool', origin='lower')
    plt.suptitle("draw_rasterized_plane")
    plt.show()


def draw_rasterized_scene(csl, box, sampling_resolution, margin):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    for plane in csl.planes:
        if not plane.is_empty:  # todo show empty planes
            mask, xyz = rasterizer_factory(plane).get_rasterazation(sampling_resolution, margin)
            ax.scatter(*xyz[mask].T)
    fig.suptitle("draw_rasterized_scene")
    plt.show()


def draw_model(network_manager, sampling_resolution=(64, 64, 64), threshold=0.5):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    x = np.linspace(-1, 1, sampling_resolution[0])
    y = np.linspace(-1, 1, sampling_resolution[1])
    z = np.linspace(-1, 1, sampling_resolution[2])

    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3))
    pred = network_manager.predict(xyz)
    label = pred > threshold
    print(f"num of dots: {len(xyz[label])} / {len(xyz)}")
    ax.scatter(*xyz[label].T)
    fig.suptitle("draw_model")
    plt.show()


def show_plane(plane):
    verts, _ = plane.pca_projected_vertices
    for component in plane.connected_components:
        plt.scatter(*verts[component.vertices_indeces_in_component].T, color='orange' if component.is_hole else 'black')
    plt.scatter([0],  [0], color='red')
    plt.suptitle("show_plane")
    plt.show()
