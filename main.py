from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from parse import parse
from OpenGLContext import testingcontext
from OpenGL.arrays import vbo
from OpenGLContext.arrays import *
from OpenGL.GL import shaders


def main():
	BaseContext = testingcontext.getInteractive()
	parse_file("csl-files/SideBishop-simplified.csl")


def OnInit(self):


def parse_file(filename):
	with open(filename, 'r') as csl_file:
		assert csl_file.readline().strip() == "CSLC"
		n_planes, lable_num = parse("{:d} {:d}", csl_file.readline().strip())
		csl_file.readline()
		for _ in range(n_planes):
			plane_id, n_verticies, n_connected_components, A, B, C, D = parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", csl_file.readline().strip())
			csl_file.readline()
			vertices = [tuple(parse("{:f} {:f} {:f}", csl_file.readline().strip())) for _ in range(n_verticies)]
			assert len(vertices) == n_verticies
			csl_file.readline()
			for _ in range(n_connected_components):
				compnent = map(int, csl_file.readline().strip().split(" "))  # todo holes
				n_vertecies_in_component = next(compnent)
				label = next(compnent)
				vertices_in_component = list(compnent)
				assert len(vertices_in_component) == n_vertecies_in_component
			pass





if __name__ == "__main__":
	main()

	#main("csl-files/SideBishop.csl")
	#main("csl-files/Brain.csl")






''' 
w,h= 500,500
def square():
    glBegin(GL_QUADS)
    glVertex2f(100, 100)
    glVertex2f(200, 100)
    glVertex2f(200, 200)
    glVertex2f(100, 200)
    glEnd()

def iterate():
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 500, 0.0, 500, 0.0, 1.0)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()

def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    iterate()
    glColor3f(1.0, 0.0, 3.0)
    square()
    glutSwapBuffers()

glutInit()
glutInitDisplayMode(GLUT_RGBA)
glutInitWindowSize(500, 500)
glutInitWindowPosition(0, 0)
wind = glutCreateWindow("OpenGL Coding Practice")
glutDisplayFunc(showScreen)
glutIdleFunc(showScreen)
glutMainLoop()
'''