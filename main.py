from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from parse import parse
import numpy as np
# from OpenGLContext import testingcontext
from OpenGL.arrays import vbo
# from OpenGLContext.arrays import *
from OpenGL.GL import shaders
import glfw
from functools import reduce


def OnInit(self):
	pass


class ConnectedComponent:
	def __init__(self, csl_file):
		compnent = map(int, csl_file.readline().strip().split(" "))  # todo holes
		self.n_vertecies_in_component = next(compnent)
		self.label = next(compnent)
		self.vertices_in_component = list(compnent)
		assert len(self.vertices_in_component) == self.n_vertecies_in_component


class Plane:
	def __init__(self, csl_file):
		self.plane_id, self.n_verticies, self.n_connected_components, A, B, C, D = \
			parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", csl_file.readline().strip())
		self.plane_params = (A, B, C, D)
		self.plane_normal = np.array([A, B, C])  # todo is normalized?
		self.plane_origin = -D * self.plane_normal
		csl_file.readline()
		self.vertices = np.array([parse("{:f} {:f} {:f}", csl_file.readline().strip()).fixed for _ in range(self.n_verticies)])
		assert len(self.vertices) == self.n_verticies
		csl_file.readline()
		self.connected_components = [ConnectedComponent(csl_file) for _ in range(self.n_connected_components)]

	def get_plane_basis(self):
		return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])  # todo

	def project_a_point(self, point):
		basis = self.get_plane_basis()
		return np.array([b @ (point - self.plane_origin) for b in basis])

	def get_projected_vertices(self):
		return [self.project_a_point(p) for p in self.vertices]


class CSL:
	def __init__(self, filename):
		with open(filename, 'r') as csl_file:
			assert csl_file.readline().strip() == "CSLC"
			self.n_planes, self.n_labels = parse("{:d} {:d}", csl_file.readline().strip())
			csl_file.readline()
			self.planes = [Plane(csl_file) for _ in range(self.n_planes)]

	def get_max_coordinat(self):
		ver = [abs(plane.vertices.max()) for plane in self.planes] + [abs(plane.vertices.min()) for plane in self.planes]
		return max(ver)



def main():

	glfw.init()
	window = glfw.create_window(800, 600, "PyOpenGL Triangle", None, None)
	glfw.set_window_pos(window, 400, 200)
	glfw.make_context_current(window)

	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)

	cs = CSL("csl-files/SideBishop.csl")
	# cs = CSL("csl-files/Brain.csl")
	#cs = CSL("csl-files/Heart-simplified.csl")
	# cs = CSL("csl-files/Heart-25-even-better.csl")
	# cs = CSL("csl-files/SideBishop-simplified.csl")
	max_coordinat = cs.get_max_coordinat()

	# setting color for background
	glClearColor(0, 0.1, 0.1, 1)

	while not glfw.window_should_close(window):
		glfw.poll_events()
		glClear(GL_COLOR_BUFFER_BIT)

		add_planes(cs, max_coordinat)
		glRotatef(0.1, 1.0, 0.0, 0.0)
		glRotatef(0.1, 0.0, 1.0, 0.0)
		glRotatef(0.1, 0.0, 0.0, 1.0)

		glfw.swap_buffers(window)

	glfw.terminate()


def add_planes(cs, max_coordinat):
	for plane in cs.planes:
		for connected_component in plane.connected_components:
			vertices = plane.vertices[connected_component.vertices_in_component]
			vertices /= max_coordinat

			v = np.array(vertices.flatten(), dtype=np.float32)
			glVertexPointer(3, GL_FLOAT, 0, v)

			colors = [1.0, 1.0, 0.0] * len(vertices)
			colors = np.array(colors, dtype=np.float32)

			glColorPointer(3, GL_FLOAT, 0, colors)

			glDrawArrays(GL_LINE_LOOP, 0, len(vertices))


if __name__ == "__main__":
	main()