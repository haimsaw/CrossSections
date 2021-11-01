import numpy as np
from matplotlib.path import Path
from abc import ABCMeta, abstractmethod
from CSL import Plane
from Helpers import *

INSIDE_LABEL = 1.0
OUTSIDE_LABEL = 0.0


def rasterizer_factory(plane: Plane):
    return EmptyPlaneRasterizer(plane) if plane.is_empty else PlaneRasterizer(plane)


class Cell:
    def __init__(self, pixel_center, label, pixel_radius, labeler, xyz_transformer):
        assert min(pixel_radius) > 0

        self.pixel_center = pixel_center
        self.label = label

        self.pixel_radius = pixel_radius

        self.labeler = labeler
        self.xyz_transformer = xyz_transformer
        self.xyz = self.xyz_transformer(np.array([self.pixel_center]))[0]

    def split_cell(self):
        new_cell_radius = self.pixel_radius / 2
        new_centers = np.array([[1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1]]) * new_cell_radius + self.pixel_center

        # its ok to use self.labeler and self.xyz_transformer since the new cells are on the same plane
        labels = self.labeler(new_centers)

        return [Cell(xy, label, new_cell_radius, self.labeler, self.xyz_transformer) for xy, label in zip(new_centers, labels)]


class IRasterizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_rasterazation_cells(self, resolution, margin, octant=None): raise NotImplementedError


class EmptyPlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert plane.is_empty
        self.csl = plane.csl
        self.plane = plane

    @property
    def vertices_boundaries(self):
        return self.csl.vertices_boundaries

    def _get_voxels(self, resolution, margin):
        d2_points = self._get_pixels(resolution, margin)

        if self.plane.plane_normal[0] != 0:
            xyzs = self.plane.get_xs(d2_points)

        elif self.plane.plane_normal[1] != 0:
            xyzs = self.plane.get_ys(d2_points)

        elif self.plane.plane_normal[2] != 0:
            xyzs = self.plane.get_zs(d2_points)

        else:
            raise Exception("invalid plane")
        return xyzs

    def _get_pixels(self, resolution, margin):
        '''

        :param resolution:
        :param margin:
        :return: samples the plane and returns coordidane representing the midpoint of the pixels and the pixel radius
        '''
        projected_vertices = self.plane.project(self.csl.all_vertices)
        top, bottom = add_margin(*get_top_bottom(projected_vertices), margin)

        # create 2d grid of pixels, in case the plane is aligned with the axes we want to ignore the dimension it is zero in
        # the pixels will be projected to the correct 3d space by xyz_transformer
        first_dim = 0 if self.plane.plane_normal[0] == 0 else 1
        second_dim = 1 if self.plane.plane_normal[0] == 0 and self.plane.plane_normal[1] == 0 else 2

        xs = np.linspace(bottom[first_dim], top[first_dim], resolution[0], endpoint=False)
        ys = np.linspace(bottom[second_dim], top[second_dim], resolution[1], endpoint=False)

        pixel_radius = np.array([xs[1] - xs[0], ys[1] - ys[0]])/2
        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)) + pixel_radius

        return xys, pixel_radius

    def get_rasterazation_cells(self, resolution, margin, octant=None):
        xys, pixel_radius = self._get_pixels(resolution, margin)

        if self.plane.plane_normal[0] != 0:
            xyz_transformer = self.plane.get_xs
        elif self.plane.plane_normal[1] != 0:
            xyz_transformer = self.plane.get_ys
        elif self.plane.plane_normal[2] != 0:
            xyz_transformer = self.plane.get_zs

        else:
            raise Exception("invalid plane")

        return list(filter(lambda cell: is_in_octant(cell.xyz, octant),
                      [Cell(xy, OUTSIDE_LABEL, pixel_radius, lambda centers: np.full(len(centers), OUTSIDE_LABEL), xyz_transformer) for xy in xys]))


class PlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert not plane.is_empty
        self.pca_projected_vertices, self.pca = plane.pca_projection  # todo should be on the plane
        self.plane = plane

    def _get_voxels(self, resolution, margin):
        '''

        :param resolution:
        :param margin:
        :return: samples the plane and returns coordidane representing the midpoint of the pixels and the pixel radius
        '''
        top, bottom = add_margin(*get_top_bottom(self.pca_projected_vertices), margin)

        xs = np.linspace(bottom[0], top[0], resolution[0], endpoint=False)
        ys = np.linspace(bottom[1], top[1], resolution[1], endpoint=False)
        pixel_radius = np.array([xs[1] - xs[0], ys[1] - ys[0]])/2

        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)) + pixel_radius
        xyzs = self.pca.inverse_transform(xys)

        return xys, xyzs, pixel_radius

    def _get_labeler(self):
        shape_vertices = []
        shape_codes = []
        hole_vertices = []
        hole_codes = []
        for component in self.plane.connected_components:
            if not component.is_hole:
                # last vertex is ignored
                shape_vertices += list(self.pca_projected_vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter
            else:
                # last vertex is ignored
                hole_vertices += list(self.pca_projected_vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter

        def labeler(xys):
            mask = Path(shape_vertices, shape_codes).contains_points(xys)
            if len(hole_vertices) > 0:
                pixels_in_hole = Path(hole_vertices, hole_codes).contains_points(xys)
                mask &= np.logical_not(pixels_in_hole)
            labels = np.where(mask, INSIDE_LABEL, np.full(mask.shape, OUTSIDE_LABEL))
            return labels
        return labeler

    def get_rasterazation_cells(self, resolution, margin, octant=None):
        xys, _, pixel_radius = self._get_voxels(resolution, margin)
        labeler = self._get_labeler()
        labels = labeler(xys)

        return list(filter(lambda cell: is_in_octant(cell.xyz, octant),
                      [Cell(xy, label, pixel_radius, labeler, self.pca.inverse_transform) for xy, label in zip(xys, labels)]))
