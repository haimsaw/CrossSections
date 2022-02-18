import numpy as np
from matplotlib.path import Path
from abc import ABCMeta, abstractmethod
from CSL import Plane
from Helpers import *
from torch.utils.data import Dataset


def domain_rasterizer_factory(plane: Plane):
    return EmptyPlaneRasterizer(plane) if plane.is_empty else PlaneRasterizer(plane)


class DomainCell:
    def __init__(self, pixel_center, pixel_radius, labeler, xyz_transformer):
        assert min(pixel_radius) > 0
        self._label = None

        self.pixel_center = pixel_center
        self.pixel_radius = pixel_radius

        self.labeler = labeler
        self.xyz = xyz_transformer(np.array([self.pixel_center]))[0]

    @property
    def label(self):
        if self._label is None:
            self._label = self._get_label()
        return self._label

    '''
    using Hoeffding's inequality we get that for the result to be in range of +- eps=0.1 from actual value w.p 1-alpha=1-0.001=99.9%
    we need 380=math.log(2/alpha)/(2*eps*eps) samples 
    '''
    def _get_label(self, accuracy=400):
        sampels = np.random.random_sample((accuracy, 2)) * self.pixel_radius + self.pixel_center
        labels = self.labeler(sampels)
        return sum(labels)/accuracy


class IRasterizer:
    __metaclass__ = ABCMeta

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

        if self.plane.normal[0] != 0:
            xyzs = self.plane.get_xs(d2_points)

        elif self.plane.normal[1] != 0:
            xyzs = self.plane.get_ys(d2_points)

        elif self.plane.normal[2] != 0:
            xyzs = self.plane.get_zs(d2_points)

        else:
            raise Exception("invalid plane")
        return xyzs

    def _get_pixels(self, resolution, margin):
        '''
        :return: samples the plane and returns coordidane representing the midpoint of the pixels and the pixel radius
        '''
        projected_vertices = self.plane.project(self.csl.all_vertices)
        top, bottom = add_margin(*get_top_bottom(projected_vertices), margin)

        # create 2d grid of pixels, in case the plane is aligned with the axes we want to ignore the dimension it is zero in
        # the pixels will be projected to the correct 3d space by xyz_transformer
        first_dim = 0 if self.plane.normal[0] == 0 else 1
        second_dim = 1 if self.plane.normal[0] == 0 and self.plane.normal[1] == 0 else 2

        xs = np.linspace(bottom[first_dim], top[first_dim], resolution[0], endpoint=False)
        ys = np.linspace(bottom[second_dim], top[second_dim], resolution[1], endpoint=False)

        pixel_radius = np.array([xs[1] - xs[0], ys[1] - ys[0]])/2
        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)) + pixel_radius

        return xys, pixel_radius

    def get_rasterazation_cells(self, resolution, margin):
        xys, pixel_radius = self._get_pixels(resolution, margin)

        if self.plane.normal[0] != 0:
            xyz_transformer = self.plane.get_xs
        elif self.plane.normal[1] != 0:
            xyz_transformer = self.plane.get_ys
        elif self.plane.normal[2] != 0:
            xyz_transformer = self.plane.get_zs

        else:
            raise Exception("invalid plane")

        return [DomainCell(xy, pixel_radius, lambda centers: np.full(len(centers), OUTSIDE_LABEL), xyz_transformer) for xy in xys]


class PlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane):
        assert not plane.is_empty
        self.pca_projected_vertices, self.pca = plane.pca_projection  # todo should be on the plane
        self.plane = plane

    def _get_voxels(self, resolution, margin):
        '''
        :return: samples the plane and returns coordidane representing the midpoint of the pixels and the pixel radius
        '''
        top, bottom = add_margin(np.array([1, 1]), np.array([-1, -1]), margin)

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
                shape_vertices += list(self.pca_projected_vertices[component.vertices_indices]) + [
                    [0, 0]]
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]
            else:
                # last vertex is ignored
                hole_vertices += list(self.pca_projected_vertices[component.vertices_indices]) + [
                    [0, 0]]
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]

        # noinspection PyTypeChecker
        def labeler(xys):
            mask = Path(shape_vertices, shape_codes).contains_points(xys)
            if len(hole_vertices) > 0:
                pixels_in_hole = Path(hole_vertices, hole_codes).contains_points(xys)
                mask &= np.logical_not(pixels_in_hole)
            labels = np.where(mask, INSIDE_LABEL, np.full(mask.shape, OUTSIDE_LABEL))
            return labels
        return labeler

    def get_rasterazation_cells(self, resolution, margin):
        xys, _, pixel_radius = self._get_voxels(resolution, margin)
        labeler = self._get_labeler()

        return [DomainCell(xy, pixel_radius, labeler, self.pca.inverse_transform) for xy in xys]


class DomainDataset(Dataset):
    def __init__(self, csl, calc_density, sampling_resolution=(256, 256), sampling_margin=0.2, transform=None, target_transform=None):
        self.csl = csl
        self.calc_density = calc_density

        cells = []
        for plane in csl.planes:
            cells += domain_rasterizer_factory(plane).get_rasterazation_cells(sampling_resolution, sampling_margin)

        self.cells = np.array(cells)

        self.xyzs = np.array([cell.xyz for cell in self.cells])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.cells.size

    def __getitem__(self, idx):
        cell = self.cells[idx]

        xyz = cell.xyz

        label = [cell.label] if self.calc_density else [-1]

        if self.transform:
            xyz = self.transform(xyz)
        if self.target_transform:
            label = self.target_transform(label)

        return xyz, label

