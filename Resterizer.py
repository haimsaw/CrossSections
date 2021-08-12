import numpy as np
from matplotlib.path import Path
from sklearn.decomposition import PCA

from CSL import Plane


def rasterizer_factory(plane: Plane):
    return EmptyPlaneRasterizer(plane) if plane.is_empty else PlaneRasterizer(plane)


class EmptyPlaneRasterizer:
    def __init__(self, plane: Plane):
        assert plane.is_empty
        self.csl = plane.csl

    @property
    def vertices_boundaries(self):
        return self.csl.vertices_boundaries

    def _get_points_to_sample(self, resolution, margin):
        top, bottom = self.vertices_boundaries
        top += margin * (top - bottom)
        bottom -= margin * (top - bottom)

        # top and bottom should be on the plane

        xx = np.linspace(bottom[0], top[0], resolution[0])
        yy = np.linspace(bottom[1], top[1], resolution[1])
        zz = np.linspace(bottom[2], top[2], resolution[1])

        xyz = np.stack(np.meshgrid(xx, yy, zz), axis=-1).reshape((-1, 2))
        return xyz

    def get_rasterazation(self, resolution, margin):
        xy, xyz = self._get_points_to_sample(resolution, margin)
        mask = self.get_rasterazation_mask(xy)
        return mask, xyz

    def get_rasterazation_mask(self, xy_flat):
        return np.full(len(xy_flat), False)


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

    def _get_points_to_sample(self, resolution, margin):
        top, bottom = self.vertices_boundaries
        top += margin * (top - bottom)
        bottom -= margin * (top - bottom)

        xvalues = np.linspace(bottom[0], top[0], resolution[0])
        yvalues = np.linspace(bottom[1], top[1], resolution[1])
        xy = np.stack(np.meshgrid(xvalues, yvalues), axis=-1).reshape((-1, 2))
        xyz = self.pca.inverse_transform(xy)
        return xy, xyz

    def get_rasterazation(self, resolution, margin):
        xy, xyz = self._get_points_to_sample(resolution, margin)
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