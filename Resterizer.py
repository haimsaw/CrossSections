import numpy as np
from matplotlib.path import Path
from abc import ABCMeta, abstractmethod
from CSL import Plane
from Helpers import *


def rasterizer_factory(plane: Plane):
    return EmptyPlaneRasterizer(plane) if plane.is_empty else PlaneRasterizer(plane)


# sample the btm left point of each cell (not the middle)
class Cell:
    def __init__(self, xy_btm_left, label, cell_size, labeler, xyz_transformer):
        assert min(cell_size) > 0

        self.xy_btm_left = xy_btm_left
        self.label = label

        self.cell_size = cell_size

        self.labeler = labeler
        self.xyz_transformer = xyz_transformer
        self.xyz = self.xyz_transformer(np.array([self.xy_btm_left]))[0]

    def split_cell(self):
        new_cell_size = self.cell_size / 2
        new_xys = np.array([[0, 0],
                            [1, 0],
                            [0, 1],
                            [1, 1]]) * new_cell_size + self.xy_btm_left

        # its ok to use self.labeler and self.xyz_transformer since the new cells are on the same plane
        labels = self.labeler(new_xys)

        return [Cell(xy, label, new_cell_size, self.labeler, self.xyz_transformer) for xy, label in zip(new_xys, labels)]


class IRasterizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_rasterazation(self, resolution, margin): raise NotImplementedError

    @abstractmethod
    def get_rasterazation_cells(self, resolution, margin): raise NotImplementedError


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
        projected_vertices = self.plane.project(self.csl.all_vertices)
        top, bottom = add_margin(*get_top_bottom(projected_vertices), margin)

        # create 2d grid of pixels, in case the plane is aligned with the axes we want to ignore the dimension it is zero in
        # the pixels will be projected to the correct 3d space by xyz_transformer
        first_dim = 0 if self.plane.plane_normal[0] == 0 else 1
        second_dim = 1 if self.plane.plane_normal[0] == 0 and self.plane.plane_normal[1] == 0 else 2

        xs = np.linspace(bottom[first_dim], top[first_dim], resolution[0])
        ys = np.linspace(bottom[second_dim], top[second_dim], resolution[1])
        xy_diffs = [xs[1] - xs[0], ys[1] - ys[0]]

        return np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)), xy_diffs

    def get_rasterazation(self, resolution, margin):
        raise NotImplementedError("should use get_rasterazation_cells instead")
        return np.full(resolution, False).reshape(-1), self._get_voxels(resolution, margin)

    def get_rasterazation_cells(self, resolution, margin):
        xys, cell_size = self._get_pixels(resolution, margin)

        if self.plane.plane_normal[0] != 0:
            xyz_transformer = self.plane.get_xs
        elif self.plane.plane_normal[1] != 0:
            xyz_transformer = self.plane.get_ys
        elif self.plane.plane_normal[2] != 0:
            xyz_transformer = self.plane.get_zs

        else:
            raise Exception("invalid plane")

        return [Cell(xy, False, cell_size, lambda x: False, xyz_transformer) for xy in xys]


class PlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert not plane.is_empty
        self.vertices, self.pca = plane.pca_projected_vertices  # todo should be on the plane
        self.plane = plane

    def _get_voxels(self, resolution, margin):
        top, bottom = add_margin(*get_top_bottom(self.vertices), margin)

        xs = np.linspace(bottom[0], top[0], resolution[0])
        ys = np.linspace(bottom[1], top[1], resolution[1])
        xy_diffs = [xs[1] - xs[0], ys[1] - ys[0]]

        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
        xyzs = self.pca.inverse_transform(xys)

        return xys, xyzs, xy_diffs

    def _get_labeler(self):
        shape_vertices = []
        shape_codes = []
        hole_vertices = []
        hole_codes = []
        for component in self.plane.connected_components:
            if not component.is_hole:
                # last vertex is ignored
                shape_vertices += list(self.vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter
            else:
                # last vertex is ignored
                hole_vertices += list(self.vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter

        def labeler(xys):
            mask = Path(shape_vertices, shape_codes).contains_points(xys)
            if len(hole_vertices) > 0:
                pixels_in_hole = Path(hole_vertices, hole_codes).contains_points(xys)
                mask &= np.logical_not(pixels_in_hole)
            return mask

        return labeler

    def get_rasterazation(self, resolution, margin):
        raise NotImplementedError("should use get_rasterazation_cells instead")

        xys, xyzs, _ = self._get_voxels(resolution, margin)
        labels = self._get_labeler()(xys)
        return labels, xyzs

    def get_rasterazation_cells(self, resolution, margin):
        xys, _, xy_diffs = self._get_voxels(resolution, margin)
        labeler = self._get_labeler()
        labels = labeler(xys)

        return [Cell(xy, label, xy_diffs, labeler, self.pca.inverse_transform) for xy, label in zip(xys, labels)]
