from itertools import chain

import numpy as np
from parse import parse
from shapely.geometry.polygon import LinearRing
from simplification.cutil import simplify_coords, simplify_coords_idx
from sklearn.decomposition import PCA
from matplotlib.path import Path

from Helpers import *

'''
CSLC
header, keep as is
6 2
number of planes, number of labels (should be at least 2 - inside and outside)

1 10 1 0.0 -1.0 0.0 -0.8
plane index (1-indexing, please state planes in order), number of vertices in the plane image (a hole is counted as another component here), number of connected components, plane parameters A,B,C,D, such that  Ax+By+Cz+D=0

0.6  -0.8 0
0.48541
-0.8 0.35267
0.18541
-0.8 0.57063
-0.18541
-0.8 0.57063
-0.48541
-0.8 0.35267
-0.6  -0.8 0
-0.48541
-0.8 -0.35267
-0.18541
-0.8 -0.57063
0.18541
-0.8 -0.57063
0.48541
-0.8 -0.35267
The vertices in x,y,z coordinates. The should be on the plane, but they are projected to the plane upon loading anyway

10 1 0 1 2 3 4 5 6 7 8 9 
image component: starts with the number of vertices, then label of the component (in case of a hole, h should be added), then the indices of vertices that form a contour of the inside label, ordered CCW.
'''


class ConnectedComponent:
    def __init__(self, parent_cc_index, label, vertices_indices):
        self.parent_cc_index = parent_cc_index
        self.label = label
        self.vertices_indices = np.array(vertices_indices)

    @classmethod
    def from_cls(cls, csl_file):
        component = iter(next(csl_file).strip().split(" "))
        sizes = next(component).split("h") + [-1]

        # parent_cc_index is the index of the ConnectedComponent in which the hole lies (applies only for holes)
        n_vertices_in_component, parent_cc_index = int(sizes[0]), int(sizes[1])
        component = map(int, component)
        label = next(component)

        # ccw for non holes and cw for holes
        vertices_indices = list(component)
        assert len(vertices_indices) == n_vertices_in_component
        return cls(parent_cc_index, label, vertices_indices)

    def __len__(self):
        return len(self.vertices_indices)

    def __repr__(self):
        return f"{len(self.vertices_indices)}{'h'+str(self.parent_cc_index) if self.is_hole else ''} {self.label} {' '.join(map(str, self.vertices_indices))}\n"

    @property
    def is_hole(self):
        return self.parent_cc_index >= 0

    @property
    def edges_indices(self):
        e1 = self.vertices_indices
        e2 = np.concatenate((self.vertices_indices[1:], self.vertices_indices[0:1]))
        return np.stack((e1, e2)).T

    @property
    def simplified_edges_indices(self):
        # todo smarter simplification
        e1 = self.vertices_indices[::2]
        e2 = np.concatenate((e1[1:], e1[0:1]))
        return np.stack((e1, e2)).T


class Plane:
    def __init__(self, plane_id: int, plane_params: tuple, vertices: np.array, connected_components: list, csl):
        assert len(plane_params) == 4

        self.csl = csl
        self.plane_id = plane_id

        self.vertices = vertices  # should be on the plane
        self.connected_components = connected_components

        # self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.plane_params = plane_params
        self.normal = np.array(plane_params[0:3])
        self.normal /= np.linalg.norm(self.normal)

        self.plane_origin = plane_origin_from_params(plane_params)

    def __repr__(self):
        plane = f"{self.plane_id} {len(self.vertices)} {len(self.connected_components)} {'{:.10f} {:.10f} {:.10f} {:.10f}'.format(*self.plane_params)}\n\n"
        verts = ''.join(['{:.10f} {:.10f} {:.10f}'.format(*vert) + '\n' for vert in self.vertices]) + "\n"
        ccs = ''.join(map(repr, self.connected_components))
        return plane + verts + ccs

    def __len__(self):
        return sum((len(cc) for cc in self.connected_components))

    def __isub__(self, point: np.array):
        assert point.shape == (3,)
        self.vertices -= point
        self.plane_origin -= point
        # todo change plane params?

        # new_D = self.plane_params[3] + np.dot(self.plane_params[:3], point)  # normal*(x-x_0)=0
        # self.plane_params = self.plane_params[:3] + (new_D,)

    def __itruediv__(self, scale: float):
        self.vertices /= scale
        # todo change plane params?

    def __imatmul__(self, rotation: PCA):
        if len(self.vertices) > 0:
            self.vertices = rotation.transform(self.vertices)
        self.plane_origin = rotation.transform([self.plane_origin])[0]
        self.normal = rotation.transform([self.normal])[0]
        # todo change plane params?

    @classmethod
    def from_csl_file(cls, csl_file, csl):
        line = next(csl_file).strip()
        plane_id, n_vertices, n_connected_components, a, b, c, d = \
            parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", line)
        plane_params = (a, b, c, d)
        vertices = np.array([parse("{:f} {:f} {:f}", next(csl_file).strip()).fixed for i in range(n_vertices)])
        if n_vertices == 0:
            vertices = np.empty(shape=(0, 3))
        assert len(vertices) == n_vertices
        connected_components = [ConnectedComponent.from_cls(csl_file) for _ in range(n_connected_components)]
        return cls(plane_id, plane_params, vertices, connected_components, csl)

    @classmethod
    def from_mesh(cls, ccs, plane_params, normal, origin, plane_id, csl):

        connected_components = []
        vertices = np.empty(shape=(0, 3))

        to_plane_cords = cls._get_to_plane_cords(ccs[0][0], normal, origin)

        for cc in ccs:
            # todo haim - this does not handles non empty holes
            if len(cc) > 2:
                is_hole, parent_cc_idx = cls._is_cc_hole(cc, ccs, to_plane_cords)

                oriented_cc = cls._orient_polyline(cc, is_hole, to_plane_cords)

                vert_start = len(vertices)
                if is_hole:
                    connected_components.append(ConnectedComponent(parent_cc_idx, 2, list(range(vert_start, vert_start+len(cc)))))
                else:
                    connected_components.append(ConnectedComponent(-1, 1, list(range(vert_start, vert_start+len(cc)))))
                vertices = np.concatenate((vertices, oriented_cc))

        return cls(plane_id, plane_params, vertices, connected_components, csl)

    @classmethod
    def _get_to_plane_cords(cls, point_on_plane, normal, origin):
        b0 = point_on_plane - origin
        b0 /= np.linalg.norm(b0)
        b1 = np.cross(normal, b0)
        transformation_matrix = np.array([b0, b1])

        def to_plane_cords(xyzs):
            alinged = xyzs - origin
            return np.array([transformation_matrix @ v for v in alinged])

        return to_plane_cords

    @classmethod
    def _orient_polyline(cls, verts, is_hole, to_plane_cords):
        if is_hole == LinearRing(to_plane_cords(verts)).is_ccw:
            oriented_verts = verts[::-1]
        else:
            oriented_verts = verts
        return oriented_verts

    @classmethod
    def _is_cc_hole(cls, cc, ccs, transform):
        is_hole = False
        parent_cc_idx = None

        point_inside_cc = transform(cc[0:1])
        for i, other_cc in enumerate(ccs):
            if other_cc is cc:
                continue
            shape_vertices = list(transform(other_cc)) + [[0, 0]]
            shape_codes = [Path.MOVETO] + [Path.LINETO] * (len(other_cc) - 1) + [Path.CLOSEPOLY]
            path = Path(shape_vertices, shape_codes)
            if path.contains_points(point_inside_cc)[0]:
                # todo not neccery 1 but enoght for my purpucess
                is_hole = True
                parent_cc_idx = i
                break
        return is_hole, parent_cc_idx

    @classmethod
    def empty_plane(cls, plane_id, plane_params, csl):
        return cls(plane_id, plane_params, np.empty(shape=(0, 3)), [], csl)

    @property
    def is_empty(self):
        return len(self.vertices) == 0

    @property
    def pca_projection(self):
        if self.is_empty:
            raise Exception("rotating empty plane")

        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(self.vertices)
        return pca.transform(self.vertices), pca

    @property
    def edges(self):
        edges = np.empty((0, 2), dtype=int)
        for cc in self.connected_components:
            edges = np.concatenate((edges, cc.edges_indices))
        return edges

    @property
    def simplified(self):
        simplified_edges = np.empty((0, 2), dtype=int)
        for cc in self.connected_components:
            simplified_edges = np.concatenate((simplified_edges, cc.simplified_edges_indices))
        new_verts = self.vertices[simplified_edges[:, 0]]
        e1 = np.arange(len(new_verts))
        e2 = np.concatenate((e1[1:], e1[0:1]))
        new_edges = np.stack((e1, e2)).T
        return new_verts, new_edges

    def simplify_RDP(self, epsilon):
        simplified_verts = np.empty((0, 3))
        ccs = []
        to_plane_cords = self._get_to_plane_cords(self.vertices[0], self.normal, self.plane_origin)

        for cc in self.connected_components:
            polygon_verts_3d = self.vertices[np.concatenate((cc.vertices_indices, cc.vertices_indices[:1]))]

            simplified_cc_idx = simplify_coords_idx(to_plane_cords(polygon_verts_3d), epsilon)[:-1]

            if len(simplified_cc_idx) > 2:
                simplified_cc_idx = self._orient_polyline(polygon_verts_3d[simplified_cc_idx], cc.is_hole, to_plane_cords)

                new_idx_start = len(simplified_verts)
                simplified_verts = np.append(simplified_verts, simplified_cc_idx, axis=0)
                ccs.append(ConnectedComponent(cc.parent_cc_index, cc.label, list(range(new_idx_start, len(simplified_verts)))))

        self.vertices = simplified_verts
        self.connected_components = ccs

    def project(self, points):
        # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
        dists = (points - self.plane_origin) @ self.normal
        return self.csl.all_vertices - np.outer(dists, self.normal)

    def get_xs(self, yzs):
        xs = (self.normal @ self.plane_origin - yzs @ self.normal[1:3]) / self.normal[0]
        xyzs = np.concatenate((xs.reshape(1, -1).T, yzs), axis=1)
        return xyzs

    def get_ys(self, xzs):
        ys = (self.normal @ self.plane_origin - xzs @ self.normal[0:3:2]) / self.normal[1]
        xyzs = np.concatenate((xzs[:, 0].reshape(1, -1).T, ys.reshape(1, -1).T, xzs[:, 1].reshape(1, -1).T), axis=1)
        return xyzs

    def get_zs(self, xys):
        zs = (self.normal @ self.plane_origin - xys @ self.normal[0:2]) / self.normal[2]
        xyzs = np.concatenate((xys, zs.reshape(1, -1).T), axis=1)
        return xyzs


class CSL:
    def __init__(self, model_name, plane_gen, n_labels):
        self.model_name = model_name
        self.n_labels = n_labels
        self.planes = plane_gen(self)

    def __len__(self):
        return sum((len(plane) for plane in self.planes))

    def __repr__(self):
        non_empty_planes = list(filter(lambda plane: len(plane.vertices) > 0, self.planes))
        return f"CSLC\n{len(non_empty_planes)} {self.n_labels} \n\n" + ''.join(map(repr, non_empty_planes))

    @classmethod
    def from_csl_file(cls, filename):
        model_name = filename.split('/')[-1].split('\\')[-1].split('.')[0]
        with open(filename, 'r') as csl_file:
            csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
            assert next(csl_file).strip() == "CSLC"
            n_planes, n_labels = parse("{:d} {:d}", next(csl_file).strip())
            plane_gen = lambda csl: [Plane.from_csl_file(csl_file, csl) for _ in range(n_planes)]
            return cls(model_name, plane_gen, n_labels)

    @classmethod
    def from_mesh(cls, model_name, plane_origins,  plane_normals, ds, ccs_per_plane):
        n_labels = 2

        def plane_gen(csl):
            planes = []
            i = 1

            for origin, normal, d, ccs in zip(plane_origins, plane_normals, ds, ccs_per_plane):
                plane_params = (*normal, d)

                if len(ccs) > 0:
                    planes.append(Plane.from_mesh(ccs, plane_params, normal, origin, i, csl))
                    i += 1
            return planes
        return cls(model_name, plane_gen, n_labels)

    @property
    def all_vertices(self):
        ver_list = (plane.vertices for plane in self.planes if not plane.is_empty)
        return list(chain(*ver_list))

    @property
    def scale_factor(self):
        return np.max(np.absolute(self.all_vertices))

    @property
    def edges_verts(self):
        scene_verts = np.empty((0, 3))
        scene_edges = np.empty((0, 2), dtype=np.int32)
        for plane in self.planes:
            plane_vert_start = len(scene_verts)

            scene_verts = np.concatenate((scene_verts, plane.vertices))
            scene_edges = np.concatenate((scene_edges, plane.edges + plane_vert_start))

        return scene_edges, scene_verts

    def to_ply(self, save_path):
        edges, verts = self.edges_verts

        header = f'ply\n' \
                 f'format ascii 1.0\n' \
                 f'element vertex {len(verts)}\n' \
                 f'property float x\nproperty float y\nproperty float z\n' \
                 f'element edge {len(edges)}\n' \
                 f'property int vertex1\n' \
                 f'property int vertex2\n' \
                 f'property uchar red\n' \
                 f'property uchar green\n' \
                 f'property uchar blue\n' \
                 f'end_header\n'

        file_name = f'{save_path}/{self.model_name}_sliced.ply'

        with open(file_name, 'w') as f:
            f.write(header)
            for v in verts:
                f.write('{:.10f} {:.10f} {:.10f}\n'.format(*v))
            for e in edges:
                f.write('{:d} {:d} 0 255 0\n'.format(*e))

    def simplify_in_place_RDP(self, epsilon):
        self.model_name += "_simplified"
        for plane in self.planes:
            if not plane.is_empty:
                plane.simplify_RDP(epsilon)

    def _add_empty_plane(self, plane_params):
        plane_id = len(self.planes) + 1
        self.planes.append(Plane.empty_plane(plane_id, plane_params, self))

    def add_boundary_planes(self, margin):
        top, bottom = add_margin(*get_top_bottom(self.all_vertices), margin)

        for i in range(3):
            normal = [0.0] * 3
            normal[i] = 1.0

            self._add_empty_plane(tuple(normal + [-top[i]]))
            self._add_empty_plane(tuple(normal + [-bottom[i]]))

        # stacked = np.stack((top, bottom))
        # return np.array([np.choose(choice, stacked) for choice in itertools.product([0, 1], repeat=3)])

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

    def scale(self, bounding_planes_margin):
        scale_factor = self.scale_factor
        if scale_factor <= 1:
            return
        for plane in self.planes:
            plane.vertices /= 1.1 * scale_factor
            plane.plane_origin /= 1.1 * scale_factor
            plane.plane_params = (*plane.normal, plane_d_from_origin(plane.plane_origin, plane.normal))

    def adjust_csl(self, bounding_planes_margin):
        # self.centralize() # todo haim meshes from csl are centralized?
        # self.rotate_by_pca()
        self.scale(bounding_planes_margin)
        self.add_boundary_planes(margin=bounding_planes_margin)
