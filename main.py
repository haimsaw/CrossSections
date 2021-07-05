from OpenGL.GL import *
from OpenGL.GLUT import *
from parse import parse
import numpy as np
import glfw
from random import random, randint
import math
import progressbar


class ConnectedComponent:
	def __init__(self, csl_file):
		component = iter(next(csl_file).strip().split(" "))  # todo holes
		sizes = next(component).split("h") + [0]

		self.n_vertices_in_component, self.n_holes = int(sizes[0]), int(sizes[1])
		component = map(int, component)
		self.label = next(component)
		#self.label = 1 if self.n_holes> 0 else 0
		self.vertices_in_component = list(component)
		assert len(self.vertices_in_component) == self.n_vertices_in_component


class Plane:
	def __init__(self, csl_file, bar):
		l = next(csl_file).strip()
		self.plane_id, self.n_vertices, self.n_connected_components, A, B, C, D = \
			parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", l)
		self.plane_params = (A, B, C, D)
		self.plane_normal = np.array([A, B, C])  # todo is normalized?
		self.plane_origin = -D * self.plane_normal
		self.vertices = np.array([parse("{:f} {:f} {:f}", next(csl_file).strip()).fixed for _ in range(self.n_vertices)])
		assert len(self.vertices) == self.n_vertices
		self.connected_components = [ConnectedComponent(csl_file) for _ in range(self.n_connected_components)]
		bar.update(self.plane_id + 1)


class CSL:
	def __init__(self, filename):
		with open(filename, 'r') as csl_file:
			csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
			assert next(csl_file).strip() == "CSLC"
			self.n_planes, self.n_labels = parse("{:d} {:d}", next(csl_file).strip())

			bar = progressbar.ProgressBar(maxval=self.n_planes+1, widgets=[progressbar.Percentage(), progressbar.Bar()])
			bar.start()

			self.planes = [Plane(csl_file, bar) for _ in range(self.n_planes)]

			bar.finish()

	def get_scale_factor(self):
		ver = [abs(plane.vertices.max()) for plane in self.planes if len(plane.vertices) > 0] + [abs(plane.vertices.min()) for plane in self.planes if len(plane.vertices) > 0]
		return max(ver)


class Renderer:
	def __init__(self, csl_file):
		self.cs = CSL(csl_file)

		self.zoom = 1

		self.rho = 0
		self.theta = 0
		self.dx = 0
		self.dy = 0

		self.colors = [[random(), random(), random()] for _ in range(self.cs.n_labels+1)]
		self.scale_factor = self.cs.get_scale_factor()

		glfw.init()
		self.window = glfw.create_window(800, 600, "Cross Sections", None, None)
		glfw.set_window_pos(self.window, 400, 200)
		glfw.make_context_current(self.window)

		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		glMatrixMode(GL_MODELVIEW)

		glfw.set_scroll_callback(self.window, self.get_on_scroll())
		glClearColor(0, 0.1, 0.1, 1)

	def get_on_scroll(self):
		def on_scroll(window, dx, dy):
			self.zoom += dy * 0.1
		return on_scroll

	def draw_scene(self):
		for plane in self.cs.planes:
			for connected_component in plane.connected_components:
				vertices = plane.vertices[connected_component.vertices_in_component]
				vertices /= self.scale_factor

				v = np.array(vertices.flatten(), dtype=np.float32)
				glVertexPointer(3, GL_FLOAT, 0, v)

				color = self.colors[connected_component.label] * len(vertices)
				color = np.array(color, dtype=np.float32)
				glColorPointer(3, GL_FLOAT, 0, color)

				glDrawArrays(GL_LINE_LOOP, 0, len(vertices))

	def event_loop(self):
		origin_x = 0
		origin_y = 0

		while not glfw.window_should_close(self.window):
			glfw.poll_events()
			glClear(GL_COLOR_BUFFER_BIT)
			glLoadIdentity()

			x, y = glfw.get_cursor_pos(self.window)

			if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
				self.rho -= (origin_x - x) / 10
				self.theta -= (origin_y - y) / 10

			elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
				self.dx -= (origin_x - x) / 100
				self.dy += (origin_y - y) / 100

			glRotatef(self.rho, 0.0, 1.0, 0.0)
			glRotatef(self.theta, 0.0, 0.0, 1.0)

			glScalef(self.zoom, self.zoom, self.zoom)

			glTranslate(self.dx, self.dy, 0)

			self.draw_scene()
			glfw.swap_buffers(self.window)

			origin_x = x
			origin_y = y

		glfw.terminate()


def main():
	# renderer = Renderer("csl-files/Heart-25-even-better.csl")
	renderer = Renderer("csl-files/Horsers.csl")
	# renderer = Renderer("csl-files/Brain.csl")
	# renderer = Renderer("csl-files/Abdomen.csl")
	# renderer = Renderer("csl-files/Vetebrae.csl")
	# renderer = Renderer("csl-files/rocker-arm.csl")
	# renderer = Renderer("csl-files/SideBishop.csl")
	# renderer = Renderer("csl-files/ParallelEight.csl")
	# renderer = Renderer("csl-files/ParallelEightMore.csl")

	renderer.event_loop()


if __name__ == "__main__":
	main()
'''
todo:
	0? draw the shape filled in the csl visualization()
	1. determine if a point is inside the box or not (cgal or google it?)
	2. add 6 empty planes on the bounding box (with +-20% margin)
	3. rasterize the plane:
		3.a pca on the points fox axis, origin in mean to get params for the plane
		3.b take -+20% of empty space
		3.c get color for reach pixel (256*256 pixels in each direction)
	4. visualize raster of the planes
'''
