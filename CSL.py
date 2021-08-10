import itertools
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
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
        self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.vertices = vertices  # todo should be on the plane
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
    def rasterizer(self):
        return PlaneRasterizer(self) if self.is_empty else EmptyPlaneRasterizer(self)

    @property
    def is_empty(self):
        return len(self.vertices) > 0

    def __isub__(self, other: np.array):
        assert len(other) == 3
        self.vertices -= other
        new_D = self.plane_params[3] + np.dot(self.plane_params[:3], other)  # normal*(x-x_0)=0
        self.plane_params = self.plane_params[:3] + (new_D,)
        return self

    def show_plane(self):
        for component in self.connected_components:
            plt.plot(*self.vertices[component.vertices_indeces_in_component].T, color='orange' if component.is_hole else 'black')
        plt.scatter([0],  [0], color='red')
        plt.show()

    def show_rasterized(self, resolution=(256, 256), margin=0.2):
        plt.imshow(self.get_rasterized(resolution, margin)[0].reshape(resolution), cmap='cool', origin='lower')
        plt.show()

    def get_rasterized(self, resolution, margin):
        return self.rasterizer.get_rasterazation(margin, resolution)


class EmptyPlaneRasterizer:
    def __init__(self, plane: Plane):
        assert plane.is_empty
        self.csl = plane.csl
        self.plane_params = plane.plane_params

    @property
    def vertices_boundaries(self):
        return self.csl.vertices_boundaries

    def get_rasterazation(self, margin, resolution):
        xy, xyz = self._get_points_to_sample(margin, resolution)
        mask = self.get_rasterazation_mask(xy)
        return mask, xyz

    def get_rasterazation_mask(self, xy_flat):
        return np.full(len(xy_flat), False)

    def _get_points_to_sample(self, margin, resolution):
        top, bottom = self.vertices_boundaries
        top += margin * (top - bottom)
        bottom -= margin * (top - bottom)
          top and bottom should be on the plane

        xx = np.linspace(bottom[0], top[0], resolution[0])
        yy = np.linspace(bottom[1], top[1], resolution[1])
        zz = np.linspace(bottom[2], top[2], resolution[1])

        xyz = np.stack(np.meshgrid(xx, yy, zz), axis=-1).reshape((-1, 2))
        return xyz

class PlaneRasterizer:
    def __init__(self, plane: Plane):
        assert not plane.is_empty
        self.pca = PCA(n_components=2, svd_solver="full")
        self.pca.fit(plane.vertices)

        self.vertices = self.pca.transform(plane.vertices)  # todo should be on the plane
        self.plane = plane

    @property
    def vertices_boundaries(self):
        top = np.amax(self.vertices, axis=0)
        bottom = np.amin(self.vertices, axis=0)
        return top, bottom

    def get_rasterazation(self, margin, resolution):
        xy, xyz = self._get_points_to_sample(margin, resolution)
        mask = self.get_rasterazation_mask(xy)
        return mask, xyz

    def get_rasterazation_mask(self, xy_flat):
        shape_vertices = []
        shape_codes = []
        hole_vertices = []
        hole_codes = []
        for component in self.plane.connected_components:
            if not component.is_hole:
                # last vertex is ignored
                shape_vertices += list(self.vertices[component.vertices_indeces_in_component]) + [
                    [0, 0]]  # todo better way?
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter
            else:
                # last vertex is ignored
                hole_vertices += list(self.vertices[component.vertices_indeces_in_component]) + [
                    [0, 0]]  # todo better way?
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter

        mask = Path(shape_vertices, shape_codes).contains_points(xy_flat)
        if len(hole_vertices) > 0:
            pixels_in_hole = Path(hole_vertices, hole_codes).contains_points(xy_flat)
            mask &= np.logical_not(pixels_in_hole)
        return mask

    def _get_points_to_sample(self, margin, resolution):
        top, bottom = self.vertices_boundaries
        top += margin * (top - bottom)
        bottom -= margin * (top - bottom)

        xvalues = np.linspace(bottom[0], top[0], resolution[0])
        yvalues = np.linspace(bottom[1], top[1], resolution[1])
        xy = np.stack(np.meshgrid(xvalues, yvalues), axis=-1).reshape((-1, 2))
        xyz = self.pca.inverse_transform(xy)
        return xy, xyz


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

    def centralize(self):
        mean = np.mean(self.all_vertices, axis=0)
        for plane in self.planes:
            plane -= mean

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

    def rotate_by_pca(self):
        all_vertices = self.all_vertices
        pca = PCA(n_components=3, svd_solver="full")
        pca.fit(all_vertices)
        for plane in self.planes:
            # todo not rotating plane params
            plane.vertices = pca.transform(plane.vertices)

    def scale(self):
        scale_factor = self.scale_factor
        for plane in self.planes:
            plane.vertices /= scale_factor
