import itertools
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from parse import parse
from sklearn.decomposition import PCA

class ConnectedComponent:
    def __init__(self, csl_file):
        component = iter(next(csl_file).strip().split(" "))
        sizes = next(component).split("h") + [-1]

        n_vertices_in_component, self.n_holes = int(sizes[0]), int(sizes[1])  # todo what is n_holes?
        component = map(int, component)
        self.label = next(component)
        # self.label = 1 if self.n_holes> 0 else 0
        self.vertices_indeces_in_component = list(component)
        assert len(self.vertices_indeces_in_component) == n_vertices_in_component

    def __len__(self):
        return len(self.vertices_indeces_in_component)

    @property
    def is_hole(self):
        return self.n_holes >= 0


class Plane:
    def __init__(self, plane_id: int, plane_params: tuple, vertices: np.array, connected_components: list, csl):
        assert len(plane_params) == 4

        self.csl = csl
        self.plane_id = plane_id

        # self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.plane_normal = np.array(plane_params[0:3])
        self.plane_normal /= np.linalg.norm(self.plane_normal)
        self.point_on_plane = np.array([-plane_params[3]/plane_params[0], 0, 0])

        self.vertices = vertices  # should be on the plane
        self.connected_components = connected_components
        self.mean = np.mean(self.vertices, axis=0) if len(self.vertices) > 0 else np.zeros((3,))

    @classmethod
    def from_csl_file(cls, csl_file, csl):
        line = next(csl_file).strip()
        plane_id, n_vertices, n_connected_components, A, B, C, D = \
            parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", line)
        plane_params = (A, B, C, D)
        vertices = np.array([parse("{:f} {:f} {:f}", next(csl_file).strip()).fixed for _ in range(n_vertices)])
        if n_vertices == 0:
            vertices = np.empty(shape=(0, 3))
        assert len(vertices) == n_vertices
        connected_components = [ConnectedComponent(csl_file) for _ in range(n_connected_components)]
        return cls(plane_id, plane_params, vertices, connected_components, csl)

    @classmethod
    def empty_plane(cls, plane_id, plane_params, csl):
        return cls(plane_id, plane_params, np.empty(shape=(0, 3)), [], csl)

    @property
    def vertices_boundaries(self):
        top = np.amax(self.vertices, axis=0)
        bottom = np.amin(self.vertices, axis=0)
        return top, bottom

    @property
    def is_empty(self):
        return len(self.vertices) == 0

    def __isub__(self, point: np.array):
        assert point.shape == (3,)
        self.vertices -= point
        self.point_on_plane -= point

        # new_D = self.plane_params[3] + np.dot(self.plane_params[:3], point)  # normal*(x-x_0)=0
        # self.plane_params = self.plane_params[:3] + (new_D,)

        return self

    def __itruediv__(self, scale: float):
        self.vertices /= scale
        self.plane_normal = self.plane_normal

    def __imatmul__(self, rotation: PCA):
        self.vertices = rotation.transform(self.vertices)
        self.point_on_plane = rotation.transform(self.point_on_plane)
        self.plane_normal = rotation.transform(self.plane_normal)


class CSL:
    def __init__(self, filename):
        with open(filename, 'r') as csl_file:
            csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
            assert next(csl_file).strip() == "CSLC"
            n_planes, self.n_labels = parse("{:d} {:d}", next(csl_file).strip())
            self.planes = [Plane.from_csl_file(csl_file, self) for _ in range(n_planes)]

    @property
    def all_vertices(self):
        ver_list = (plane.vertices for plane in self.planes if not plane.is_empty)
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
        self.planes.append(Plane.empty_plane(plane_id, plane_params, self))

    def add_boundary_planes(self, margin):
        top, bottom = self.vertices_boundaries

        top += margin * (top - bottom)
        bottom -= margin * (top - bottom)

        for i in range(3):
            normal = [0] * 3
            normal[i] = 1

            self.__add_empty_plane(tuple(normal + [top[i]]))
            self.__add_empty_plane(tuple(normal + [bottom[i]]))

        stacked = np.stack((top, bottom))
        return np.array([np.choose(choice, stacked) for choice in itertools.product([0, 1], repeat=3)])

    def centralize(self):
        mean = np.mean(self.all_vertices, axis=0)
        for plane in self.planes:
            plane -= mean

    def rotate_by_pca(self):
        all_vertices = self.all_vertices
        pca = PCA(n_components=3, svd_solver="full")
        pca.fit(all_vertices)
        for plane in self.planes:
            plane @= pca

    def scale(self):
        scale_factor = self.scale_factor
        for plane in self.planes:
            plane.vertices /= scale_factor

