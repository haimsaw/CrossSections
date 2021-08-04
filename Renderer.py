from math import radians
from random import random

import glfw
import numpy as np
#from OpenGL.GL import *
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

'''
class Renderer:
    def __init__(self, csl, box):
        self.csl = csl

        self.cursor_origin = np.zeros(2, dtype='float64')

        self.rotation = Quaternion(axis=(1, 0, 0), radians=0)
        self.translation = np.zeros(2, dtype='float64')
        self.zoom = 0.5

        self.colors = [[random(), random(), random()] for _ in range(self.csl.n_labels + 1)]
        self.scale_factor = self.csl.scale_factor
        self.box = box / self.scale_factor

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
                vertices /= self.scale_factor
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
'''


class Renderer2:
    def __init__(self, csl, box):
        self.csl = csl

        self.colors = [[random(), random(), random()] for _ in range(self.csl.n_labels + 1)]
        self.scale_factor = self.csl.scale_factor
        self.box = box / self.scale_factor

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = plt.axes(projection='3d')

    def draw_scene(self):
        for plane in self.csl.planes:
            for connected_component in plane.connected_components:
                vertices = plane.vertices[connected_component.vertices_indeces_in_component]
                vertices /= self.csl.scale_factor
                alpha = 1 if connected_component.is_hole else 0.1
                self.ax.plot_trisurf(*vertices.T, color=self.colors[connected_component.label], alpha=alpha)
        # todo show box
        self.ax.plot_trisurf(*self.box.T, color=self.colors[-1])

        plt.show()
        # self.__draw_vertices(self.box, self.csl.n_labels, GL_LINE_LOOP)

    def draw_rasterized_scene(self, sampling_resolution, margin):
        top, bottom = self.csl.vertices_boundaries

        for plane in self.csl.planes:
            if len(plane.vertices) > 0:
                mask, xyz = plane.rasterizer.get_rasterized(sampling_resolution, margin)
                self.ax.scatter(*xyz[mask].T, color=self.colors[0])
        plt.show()
