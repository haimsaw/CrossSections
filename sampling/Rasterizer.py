from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib.path import Path

from hp import args
from sampling.CSL import Plane
from Helpers import add_margin, get_top_bottom
from sampling.Cell import Cell

INSIDE_LABEL = 0.0
OUTSIDE_LABEL = 1.0


def slices_rasterizer_factory(plane: Plane, hp):
    return EmptyPlaneRasterizer(plane, hp) if plane.is_empty else PlaneRasterizer(plane, hp)


class Labler:
    def __init__(self, path, hole_path):
        self.path = path
        self.hole_path = hole_path

    def __call__(self, xys):
        if self.path is None:
            return np.full(len(xys), OUTSIDE_LABEL)
        mask = self.path.contains_points(xys)
        # todo haim - this does not handles non empty holes
        if self.hole_path is not None:
            pixels_in_hole = self.hole_path.contains_points(xys)
            mask &= np.logical_not(pixels_in_hole)
        labels = np.where(mask, np.full(mask.shape, INSIDE_LABEL), np.full(mask.shape, OUTSIDE_LABEL))
        return labels


class IRasterizer:
    __metaclass__ = ABCMeta

    def __init__(self, hp):
        self.hp = hp

    @abstractmethod
    def get_rasterazation_cells(self, gen): raise NotImplementedError


class EmptyPlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane, hp):
        super().__init__(hp)
        assert plane.is_empty
        self.csl = plane.csl
        self.plane = plane

    @property
    def vertices_boundaries(self):
        return self.csl.vertices_boundaries

    def _get_voxels(self):
        d2_points = self._get_pixels()

        if self.plane.normal[0] != 0:
            xyzs = self.plane.get_xs(d2_points)

        elif self.plane.normal[1] != 0:
            xyzs = self.plane.get_ys(d2_points)

        elif self.plane.normal[2] != 0:
            xyzs = self.plane.get_zs(d2_points)

        else:
            raise Exception("invalid plane")
        return xyzs

    def _get_pixels(self):
        '''
        :return: samples the plane and returns coordidane representing the midpoint of the pixels and the pixel radius
        '''
        resolution = self.hp.root_sampling_resolution_2d
        margin = self.hp.sampling_margin

        projected_vertices = self.plane.project(self.csl.all_vertices)
        top, bottom = add_margin(*get_top_bottom(projected_vertices), margin)

        # create 2d grid of pixels, in case the plane is aligned with the axes we want to ignore the dimension it is zero in
        # the pixels will be projected to the correct 3d space by xyz_transformer
        first_dim = 0 if self.plane.normal[0] == 0 else 1
        second_dim = 1 if self.plane.normal[0] == 0 and self.plane.normal[1] == 0 else 2

        xs = np.linspace(bottom[first_dim], top[first_dim], resolution[0], endpoint=False)
        ys = np.linspace(bottom[second_dim], top[second_dim], resolution[1], endpoint=False)

        pixel_radius = np.array([xs[1] - xs[0], ys[1] - ys[0]]) / 2
        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2)) + pixel_radius

        return xys, pixel_radius

    def get_rasterazation_cells(self, gen):
        xys, pixel_radius = self._get_pixels()

        if self.plane.normal[0] != 0:
            xyz_transformer = self.plane.get_xs
        elif self.plane.normal[1] != 0:
            xyz_transformer = self.plane.get_ys
        elif self.plane.normal[2] != 0:
            xyz_transformer = self.plane.get_zs

        else:
            raise Exception("invalid plane")

        labler = Labler(None, None)
        return [Cell(xy, labler, xyz_transformer, -1) for xy in xys]


class PlaneRasterizer(IRasterizer):
    def __init__(self, plane: Plane, hp):
        super().__init__(hp)
        assert not plane.is_empty
        self.pca_projected_vertices, self.pca = plane.pca_projection  # todo should be on the plane
        self.plane = plane

    def _get_voxels(self, gen):
        radius = 0.5 ** args.sampling_radius_exp[gen]
        n_samples = args.n_samples[gen]

        xys_around_edges, xys_on_edge = self.sample_around_edge(n_samples, radius)

        thetas = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
        points_on_unit_spere = radius * np.stack((np.cos(thetas), np.sin(thetas))).T

        xys_around_vert = np.empty((0, 2))
        xyz_on_vert = self.pca_projected_vertices

        for vert in xyz_on_vert:
            xys_around_vert = np.concatenate((xys_around_vert, points_on_unit_spere + vert))

        noise = 2 * np.random.random_sample((args.n_white_noise, 2)) - 1

        if args.should_perturb_samples:
            xys_around_edges, xys_around_vert = self.perturb_samples(radius, xys_around_edges, xys_around_vert)

        # no nees to return xys_on_vert, it's contained on xys_on_edge
        return np.concatenate((xys_around_vert, xys_around_edges, noise)), xys_on_edge

    def sample_around_edge(self, n_samples, radius):
        edges_2d = self.pca_projected_vertices[self.plane.edges]
        edges_directions = edges_2d[:, 0, :] - edges_2d[:, 1, :]
        edge_normals = edges_directions @ np.array([[0, 1], [-1, 0]])
        edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]
        dist = np.linspace(0, 1, n_samples, endpoint=False)
        xys_around_edges = np.empty((0, 2))
        xys_on_edge = np.empty((0, 2))
        for edge, normal in zip(edges_2d, edge_normals):
            points_on_edge = np.array([d * edge[0] + (1 - d) * edge[1] for d in dist])

            xys_around_edges = np.concatenate(
                (xys_around_edges, points_on_edge + normal * radius, points_on_edge - normal * radius))
            xys_on_edge = np.concatenate((xys_on_edge, points_on_edge))
        return xys_around_edges, xys_on_edge

    def perturb_samples(self, radius, xys_around_edges, xys_around_vert):
        perturbation = np.random.randn(2, len(xys_around_vert))
        perturbation /= np.linalg.norm(perturbation, axis=0)
        perturbation = perturbation.T * (radius / 2)
        xys_around_vert += perturbation
        perturbation = np.random.randn(2, len(xys_around_edges))
        perturbation /= np.linalg.norm(perturbation, axis=0)
        perturbation = perturbation.T * (radius / 2)
        xys_around_edges += perturbation
        return xys_around_edges, xys_around_vert

    def _get_labeler(self):
        shape_vertices = []
        shape_codes = []
        hole_vertices = []
        hole_codes = []
        for component in self.plane.connected_components:
            if not component.is_hole:
                # last vertex is ignored
                shape_vertices += list(self.pca_projected_vertices[component.vertices_indices]) + [[0, 0]]
                shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]
            else:
                # last vertex is ignored
                hole_vertices += list(self.pca_projected_vertices[component.vertices_indices]) + [[0, 0]]
                hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]

        # noinspection PyTypeChecker
        path = Path(shape_vertices, shape_codes)
        hole_path = Path(hole_vertices, hole_codes) if len(hole_vertices) > 0 else None

        return Labler(path, hole_path)

    def get_rasterazation_cells(self, gen):
        xys_around_conture, xys_on_conture = self._get_voxels(gen)
        labeler = self._get_labeler()
        cells = [Cell(xy, labeler, self.pca.inverse_transform, self.plane.plane_id) for xy in xys_around_conture] + \
                [Cell(xy, labeler, self.pca.inverse_transform, self.plane.plane_id, is_on_edge=True) for xy in xys_on_conture]

        return cells
