from parse import parse


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
		csl_file.readline()
		self.vertices = [tuple(parse("{:f} {:f} {:f}", csl_file.readline().strip())) for _ in range(self.n_verticies)]
		assert len(self.vertices) == self.n_verticies
		csl_file.readline()
		self.connected_components = [ConnectedComponent(csl_file) for _ in range(self.n_connected_components)]


class CSL:
	def __init__(self, filename):
		with open(filename, 'r') as csl_file:
			assert csl_file.readline().strip() == "CSLC"
			self.n_planes, self.n_labels = parse("{:d} {:d}", csl_file.readline().strip())
			csl_file.readline()
			self.planes = [Plane(csl_file) for _ in range(self.n_planes)]


def main():
	# BaseContext = testingcontext.getInteractive()
	# cs = CSL("csl-files/SideBishop.csl")
	# cs = CSL("csl-files/Brain.csl")

	cs = CSL("csl-files/SideBishop-simplified.csl")
	pass


if __name__ == "__main__":
	main()

