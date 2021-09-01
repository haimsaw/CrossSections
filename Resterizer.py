import numpy as np
from matplotlib.path import Path
from abc import ABCMeta, abstractmethod
from CSL import Plane
import Helpers


def rasterizer_factory(plane: Plane):
    return EmptyPlaneRasterizer(plane) if plane.is_empty else PlaneRasterizer(plane)


# todo - sample the btm left point of each cell (not the middle)
class Cell:
    def __init__(self, xy_btm_left, cell_size, label, xyz_transformer):
        self.xy_btm_left = xy_btm_left
        self.cell_size = cell_size
        self.label = label
        self.xyz_transformer = xyz_transformer
        self.xyz = self.xyz_transformer([self.xy_btm_left])


class IRasterizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_rasterazation(self, resolution, margin): raise NotImplementedError

    #@abstractmethod
    #def get_rasterazation_cells(self, resolution, margin): raise NotImplementedError


class EmptyPlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert plane.is_empty
        self.csl = plane.csl
        self.plane = plane

    @property
    def vertices_boundaries(self):
        return self.csl.vertices_boundaries

    def _get_voxels_edges(self, resolution, margin):
        projected_vertices = self.plane.project(self.csl.all_vertices)
        top, bottom = Helpers.add_margin(*Helpers.get_top_bottom(projected_vertices), margin)

        if self.plane.plane_normal[0] != 0:
            ys = np.linspace(bottom[1], top[1], resolution[0])
            zs = np.linspace(bottom[2], top[2], resolution[1])
            xyzs = self.plane.get_xs(np.stack(np.meshgrid(ys, zs), axis=-1).reshape((-1, 2)))

        elif self.plane.plane_normal[1] != 0:
            xs = np.linspace(bottom[0], top[0], resolution[0])
            zs = np.linspace(bottom[2], top[2], resolution[1])
            xyzs = self.plane.get_ys(np.stack(np.meshgrid(xs, zs), axis=-1).reshape((-1, 2)))

        elif self.plane.plane_normal[2] != 0:
            xs = np.linspace(bottom[0], top[0], resolution[0])
            ys = np.linspace(bottom[1], top[1], resolution[1])
            xyzs = self.plane.get_zs(np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)))

        else:
            raise Exception("invalid plane")

        return xyzs

    def get_rasterazation(self, resolution, margin):
        return np.full(resolution, False).reshape(-1), self._get_voxels_edges(resolution, margin)


class PlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert not plane.is_empty
        self.vertices, self.pca = plane.pca_projected_vertices  # todo should be on the plane
        self.plane = plane

    def _get_voxels_edges(self, resolution, margin):
        top, bottom = Helpers.add_margin(*Helpers.get_top_bottom(self.vertices), margin)

        xs = np.linspace(bottom[0], top[0], resolution[0])
        ys = np.linspace(bottom[1], top[1], resolution[1])
        xy_diffs = [xs[1] - xs[0], ys[1] - ys[0]]

        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
        xyzs = self.pca.inverse_transform(xys)

        return xys, xyzs, xy_diffs

    def _get_rasterazation_mask(self, xys):
        shape_vertices = []
        shape_codes = []
        hole_vertices = []
        hole_codes = []
        for component in self.plane.connected_components:
            if not component.is_hole:
                # last vertex is ignored
                shape_vertices += list(self.vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]  # todo better way?
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter
            else:
                # last vertex is ignored
                hole_vertices += list(self.vertices[component.vertices_indices_in_component]) + [
                    [0, 0]]  # todo better way?
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]  # todo iter

        mask = Path(shape_vertices, shape_codes).contains_points(xys)
        if len(hole_vertices) > 0:
            pixels_in_hole = Path(hole_vertices, hole_codes).contains_points(xys)
            mask &= np.logical_not(pixels_in_hole)
        return mask

    def get_rasterazation(self, resolution, margin):
        xys, xyzs, _ = self._get_voxels_edges(resolution, margin)
        mask = self._get_rasterazation_mask(xys)
        return mask, xyzs

    def get_rasterazation_cells(self,  resolution, margin):
        xys, xyzs, xy_diffs = self._get_voxels_edges(resolution, margin)
        mask = self._get_rasterazation_mask(xys)

        return [Cell(xy, xy_diffs, label, self.pca.inverse_transform) for xy, label in zip(xys, mask)]