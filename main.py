import itertools

from OpenGL.GL import *
from OpenGL.GLUT import *
from parse import parse
from random import random, randint
from math import radians, degrees, acos, cos, sin
from itertools import chain
# from quaternion import as_vector_part, from_float_array
from pyquaternion import Quaternion
import numpy as np
import glfw
import progressbar


class ConnectedComponent:
    def __init__(self, csl_file):
        component = iter(next(csl_file).strip().split(" "))  # todo holes
        sizes = next(component).split("h") + [0]

        n_vertices_in_component, self.n_holes = int(sizes[0]), int(sizes[1])  # todo what is n_holes?
        component = map(int, component)
        self.label = next(component)
        # self.label = 1 if self.n_holes> 0 else 0
        self.vertices_in_component = list(component)
        assert len(self.vertices_in_component) == n_vertices_in_component


class Plane:
    def __init__(self, plane_id: int, plane_params: tuple, vertices: np.array, connected_components: list):
        assert len(plane_params) == 4

        self.plane_id = plane_id
        self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.vertices = vertices  # todo should be on the plane
        self.connected_components = connected_components

    @classmethod
    def from_csl_file(cls, csl_file, bar):
        line = next(csl_file).strip()
        plane_id, n_vertices, n_connected_components, A, B, C, D = \
            parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", line)
        plane_params = (A, B, C, D)
        vertices = np.array([parse("{:f} {:f} {:f}", next(csl_file).strip()).fixed for _ in range(n_vertices)])
        assert len(vertices) == n_vertices
        connected_components = [ConnectedComponent(csl_file) for _ in range(n_connected_components)]
        bar.update(plane_id + 1)
        return cls(plane_id, plane_params, vertices, connected_components)

    @classmethod
    def empty_plane(cls, plane_id, plane_params):
        return cls(plane_id, plane_params, np.array([]), [])

    def __isub__(self, other: np.array):
        assert len(other) == 3
        self.vertices -= other
        new_D = self.plane_params[3] + np.dot(self.plane_params[:3], other)  # normal*(x-x_0)=0
        self.plane_params = self.plane_params[:3] + (new_D,)
        return self


class CSL:
    def __init__(self, filename):
        with open(filename, 'r') as csl_file:
            csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
            assert next(csl_file).strip() == "CSLC"
            n_planes, self.n_labels = parse("{:d} {:d}", next(csl_file).strip())

            bar = progressbar.ProgressBar(maxval=n_planes + 1, widgets=[progressbar.Percentage(), progressbar.Bar()])
            bar.start()

            self.planes = [Plane.from_csl_file(csl_file, bar) for _ in range(n_planes)]

            bar.finish()

    @property
    def all_vertices(self):
        ver_list = (plane.vertices for plane in self.planes if len(plane.vertices) > 0)
        return list(chain(*ver_list))

    @property
    def scale_factor(self):
        return np.max(self.all_vertices)

    @property
    def vertices_boundaries(self):
        vertices = self.all_vertices
        top = np.amax(vertices, axis=0)
        bottom = np.amin(vertices, axis=0)
        return top, bottom

    def __add_empty_plane(self, plane_params):
        plane_id = len(self.planes) + 1
        self.planes.append(Plane.empty_plane(plane_id, plane_params))

    def centralize(self):
        mean = np.mean(self.all_vertices, axis=0)
        for plane in self.planes:
            plane -= mean

    def add_boundary_planes(self, margin_percent):
        top, bottom = self.vertices_boundaries
        margin = margin_percent * (top - bottom)

        top += margin
        bottom -= margin

        for i in range(3):
            normal = [0] * 3
            normal[i] = 1

            self.__add_empty_plane(tuple(normal + [top[i]]))
            self.__add_empty_plane(tuple(normal + [bottom[i]]))

        stacked = np.stack((top, bottom))
        return np.array([np.choose(choice, stacked) for choice in itertools.product([0, 1], repeat=3)])


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
        glfw.set_window_pos(self.window, 400, 200)
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
                vertices = plane.vertices[connected_component.vertices_in_component]
                vertices /= self.scale_factor
                self.__draw_vertices(vertices, connected_component.label)
        self.__draw_vertices(self.box, self.csl.n_labels)

    def __draw_vertices(self, vertices: np.array, label):
        v = np.array(vertices.flatten(), dtype=np.float32)
        glVertexPointer(3, GL_FLOAT, 0, v)
        color = self.colors[label] * len(vertices)
        color = np.array(color, dtype=np.float32)
        glColorPointer(3, GL_FLOAT, 0, color)
        glDrawArrays(GL_LINE_LOOP, 0, len(vertices))

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


def main():
    csl = CSL("csl-files/Heart-25-even-better.csl")
    # csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/Brain.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/rocker-arm.csl")
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")

    csl.centralize()
    box = csl.add_boundary_planes(0.2)

    renderer = Renderer(csl, box)
    renderer.event_loop()


if __name__ == "__main__":
    main()
'''
todo:
	0? draw the shape filled in the csl visualization()
	1. determine if a point is inside the box or not (cgal or google it?)
	3. rasterize the plane:
		3.a pca on the points fox axis, origin in mean to get params for the plane
		3.b take -+20% of empty space
		3.c get color for reach pixel (256*256 pixels in each direction)
	4. visualize raster of the planes
'''
