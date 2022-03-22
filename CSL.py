from itertools import chain

from parse import parse
from sklearn.decomposition import PCA

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
    def __init__(self, csl_file):
        component = iter(next(csl_file).strip().split(" "))
        sizes = next(component).split("h") + [-1]

        # parent_cc_index is the index of the ConnectedComponent in which the hole lies (applies only for holes)
        n_vertices_in_component, self.parent_cc_index = int(sizes[0]), int(sizes[1])
        component = map(int, component)
        self.label = next(component)

        # ccw for non holes and cw for holes
        self.vertices_indices = list(component)
        assert len(self.vertices_indices) == n_vertices_in_component

    def __len__(self):
        return len(self.vertices_indices)

    def __repr__(self):
        return f"{len(self.vertices_indices)}{'h'+str(self.parent_cc_index) if self.is_hole else ''} {self.label} {' '.join(map(str, self.vertices_indices))}\n"

    @property
    def is_hole(self):
        return self.parent_cc_index >= 0


class Plane:
    def __init__(self, plane_id: int, plane_params: tuple, vertices: np.array, connected_components: list, csl):
        assert len(plane_params) == 4

        self.csl = csl
        self.plane_id = plane_id

        self.vertices = vertices  # should be on the plane
        self.connected_components = connected_components
        self.mean = np.mean(self.vertices, axis=0) if len(self.vertices) > 0 else np.zeros((3,))

        # self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.plane_params = plane_params
        self.normal = np.array(plane_params[0:3])
        self.normal /= np.linalg.norm(self.normal)

        if plane_params[0] != 0:
            self.plane_origin = np.array([-plane_params[3] / plane_params[0], 0, 0])
        elif plane_params[1] != 0:
            self.plane_origin = np.array([0, -plane_params[3] / plane_params[1], 0])
        else:
            self.plane_origin = np.array([0, 0, -plane_params[3] / plane_params[2]])

    def __repr__(self):
        plane = f"{self.plane_id} {len(self.vertices)} {len(self.connected_components)} {' '.join(map(str, self.plane_params))}\n\n"
        verts = ''.join([' '.join(map(str, vert)) + '\n' for vert in self.vertices]) + "\n\n"
        ccs = ''.join(map(repr, self.connected_components))
        return plane + verts + ccs

    def __len__(self):
        return sum((len(cc) for cc in self.connected_components))

    def __isub__(self, point: np.array):
        assert point.shape == (3,)
        self.vertices -= point
        self.plane_origin -= point

        # new_D = self.plane_params[3] + np.dot(self.plane_params[:3], point)  # normal*(x-x_0)=0
        # self.plane_params = self.plane_params[:3] + (new_D,)

    def __itruediv__(self, scale: float):
        self.vertices /= scale
        # todo change plane params?

    def __imatmul__(self, rotation: PCA):
        self.vertices = rotation.transform(self.vertices)
        self.plane_origin = rotation.transform([self.plane_origin])[0]
        self.normal = rotation.transform([self.normal])[0]

    @classmethod
    def from_csl_file(cls, csl_file, csl):
        line = next(csl_file).strip()
        plane_id, n_vertices, n_connected_components, a, b, c, d = \
            parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", line)
        plane_params = (a, b, c, d)
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
    def is_empty(self):
        return len(self.vertices) == 0

    @property
    def pca_projection(self):
        if self.is_empty:
            raise Exception("rotating empty plane")

        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(self.vertices)
        return pca.transform(self.vertices), pca

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
    def __init__(self, filename):
        self.model_name = filename.split('/')[-1].split('.')[0]
        with open(filename, 'r') as csl_file:
            csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
            assert next(csl_file).strip() == "CSLC"
            n_planes, self.n_labels = parse("{:d} {:d}", next(csl_file).strip())
            self.planes = [Plane.from_csl_file(csl_file, self) for _ in range(n_planes)]

    def __len__(self):
        return sum((len(plane) for plane in self.planes))

    def __repr__(self):
        non_empty_planes = list(filter(lambda plane: len(plane.vertices) > 0, self.planes))
        return f"CSLC\n{len(non_empty_planes)} {self.n_labels} \n\n" + ''.join(map(repr, non_empty_planes))

    @property
    def all_vertices(self):
        ver_list = (plane.vertices for plane in self.planes if not plane.is_empty)
        return list(chain(*ver_list))

    @property
    def scale_factor(self):
        return np.max(np.absolute(self.all_vertices))

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
        for plane in self.planes:
            plane.vertices /= scale_factor
            # scale again so that the bounding planes will be in the range
            plane.vertices *= (1 - 2 * bounding_planes_margin)

    def adjust_csl(self, bounding_planes_margin):
        self.centralize()
        self.rotate_by_pca()
        self.scale(bounding_planes_margin)
        self.add_boundary_planes(margin=bounding_planes_margin)
