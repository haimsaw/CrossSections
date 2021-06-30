from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from parse import parse
import numpy as np
import glfw
from random import random, randint
import math


class ConnectedComponent:
	def __init__(self, csl_file):
		component = map(int, csl_file.readline().strip().split(" "))  # todo holes
		self.n_vertices_in_component = next(component)
		self.label = next(component)
		#self.label = randint(0,19)
		self.vertices_in_component = list(component)
		assert len(self.vertices_in_component) == self.n_vertices_in_component


class Plane:
	def __init__(self, csl_file):
		self.plane_id, self.n_vertices, self.n_connected_components, A, B, C, D = \
			parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", csl_file.readline().strip())
		self.plane_params = (A, B, C, D)
		self.plane_normal = np.array([A, B, C])  # todo is normalized?
		self.plane_origin = -D * self.plane_normal
		csl_file.readline()
		self.vertices = np.array([parse("{:f} {:f} {:f}", csl_file.readline().strip()).fixed for _ in range(self.n_vertices)])
		assert len(self.vertices) == self.n_vertices
		csl_file.readline()
		self.connected_components = [ConnectedComponent(csl_file) for _ in range(self.n_connected_components)]


class CSL:
	def __init__(self, filename):
		with open(filename, 'r') as csl_file:
			assert csl_file.readline().strip() == "CSLC"
			self.n_planes, self.n_labels = parse("{:d} {:d}", csl_file.readline().strip())
			csl_file.readline()
			self.planes = [Plane(csl_file) for _ in range(self.n_planes)]

	def get_scale_factor(self):
		ver = [abs(plane.vertices.max()) for plane in self.planes] + [abs(plane.vertices.min()) for plane in self.planes]
		return max(ver)


def draw_scene(cs, scale_factor, colors):
	for plane in cs.planes:
		for connected_component in plane.connected_components:
			vertices = plane.vertices[connected_component.vertices_in_component]
			vertices /= scale_factor

			v = np.array(vertices.flatten(), dtype=np.float32)
			glVertexPointer(3, GL_FLOAT, 0, v)

			color = colors[connected_component.label] * len(vertices)
			color = np.array(color, dtype=np.float32)
			glColorPointer(3, GL_FLOAT, 0, color)

			glDrawArrays(GL_LINE_LOOP, 0, len(vertices))


class Trackball:
	def __init__(self):
		self.radius = 1

	def zoom(self, distance):
		self.radius += distance

	def get_on_scroll(self):
		def on_scroll(window, dx, dy):
			self.zoom(dy * 0.1)
		return on_scroll

	def reset(self):
		self.radius = 1



def main():
	glfw.init()
	window = glfw.create_window(800, 600, "Cross Sections", None, None)
	glfw.set_window_pos(window, 400, 200)
	glfw.make_context_current(window)

	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	glMatrixMode(GL_PROJECTION)

	trackball = Trackball()  # todo remove Trackball

	# cs = CSL("csl-files/SideBishop.csl")
	cs = CSL("csl-files/Heart-25-even-better.csl")

	# cs = CSL("csl-files/ParallelEight.csl")
	# cs = CSL("csl-files/ParallelEightMore.csl")

	colors = [[random(), random(), random()] for _ in range(cs.n_labels)]
	scale_factor = cs.get_scale_factor()

	# setting color for background
	glClearColor(0, 0.1, 0.1, 1)

	glfw.set_scroll_callback(window, trackball.get_on_scroll())

	origin_x = 0
	origin_y = 0

	while not glfw.window_should_close(window):
		glfw.poll_events()
		glClear(GL_COLOR_BUFFER_BIT)

		glScalef(trackball.radius, trackball.radius, trackball.radius)

		x, y = glfw.get_cursor_pos(window)

		if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
			glRotatef((origin_x - x) / 10, 0.0, 1.0, 0.0)
			glRotatef(-(origin_y - y) / 10, 1.0, 0.0, 0.0)

		elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
			glTranslate(-(origin_x - x) / 100, (origin_y - y) / 100, 0)

		origin_x = x
		origin_y = y

		draw_scene(cs, scale_factor, colors)

		trackball.reset()
		glfw.swap_buffers(window)

	glfw.terminate()


if __name__ == "__main__":
	main()
