import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import SubsetRandomSampler

from NetManager import *
from itertools import combinations


class OctNode:
    def __init__(self, csl, center, parent, radius, oct_overlap_margin, path, **haimnet_kwargs):
        self.center = center

        self.csl = csl
        self.radius = radius
        self.oct_overlap_margin = oct_overlap_margin
        self.branches = None
        self.is_leaf = True
        self.path = path

        res = None if parent is None else parent.haim_net_manager.module
        self.haim_net_manager = HaimNetManager(csl, residual_module=res, octant=self.oct, **haimnet_kwargs)

    def __str__(self):
        return f"pos: {self.center}, radius: {self.radius}, path={self.path}"

    @staticmethod
    def _find_branch(root, position):
        """
        helper function
        returns an index corresponding to a branch
        pointing in the direction we want to go
        """
        index = 0
        if position[0] >= root.position[0]:
            index |= 4
        if position[1] >= root.position[1]:
            index |= 2
        if position[2] >= root.position[2]:
            index |= 1
        return index

    @property
    def _directions(self):
        dirs = np.array(((-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
                         (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1)))
        dirs.flags.writeable = False
        return dirs

    @property
    def depth(self):
        return len(self.path)

    @property
    def oct(self):
        radius_with_margin = self.radius * (1 + 2 * self.oct_overlap_margin)
        return np.stack((self.center + radius_with_margin, self.center - radius_with_margin))

    @property
    def oct_core(self):
        return np.stack((self.center + self.radius, self.center - self.radius))

    @property
    def _boundaries_per_direction(self):
        # +1 for boundary in the pos dir, -1 neg dir, 0 non boundary
        return (np.sum(np.array(self.path).T, axis=1) / self.depth).astype(int)

    @property
    def _vertices(self):

        # each branch corresponds to a vertex in the same direction
        vertices_directions = self._directions

        vertices = np.array([self.center + direction * self.radius for direction in vertices_directions])

        # if a certain vertices_directions has a dimension aligned with boundaries_per_direction the vertex
        # corresponding to this direction is on the boundary
        is_on_boundary = [np.any(np.equal(direction, self._boundaries_per_direction)) for direction in vertices_directions]
        return vertices, is_on_boundary

    @property
    def _edges(self):
        if self.depth == 0:
            return []

        vertices, vertices_is_on_boundary = self._vertices
        verts_directions = self._directions

        edges = list(combinations(vertices, 2))

        # consider only edges on adjacent vertices (no diagonals)
        # edge is not a diagonal if its vertices agree in two coordinates
        legal_edges = [sum(d1 * d2) == 1 for d1, d2 in combinations(verts_directions, 2)]

        edges = [edge for edge, is_legal in zip(edges, legal_edges) if is_legal]
        is_edge_verts_on_boundary = [(b1, b2) for (b1, b2), is_legal in zip(combinations(vertices_is_on_boundary, 2), legal_edges) if is_legal]

        return edges, is_edge_verts_on_boundary

    @property
    def _faces(self):
        if self.depth == 0:
            return []

        vertices, vertices_is_on_boundary = self._vertices
        verts_directions = self._directions

        faces = list(combinations(vertices, 4))

        # consider only legal combinations of verts - faces on adjacent vertices
        # faces is legal if its vertices agree on one coordinates
        legal_faces = [max(np.abs(np.sum(dirs, axis=0))) == 4 for dirs in combinations(verts_directions, 4)]

        faces = [edge for edge, is_legal in zip(faces, legal_faces) if is_legal]
        is_face_verts_on_boundary = [(b1, b2, b3, b4) for (b1, b2, b3, b4), is_legal in zip(combinations(vertices_is_on_boundary, 4), legal_faces) if is_legal]

        return faces, is_face_verts_on_boundary

    @property
    def _overlap_radius(self):
        return 2 * self.radius * self.oct_overlap_margin

    def indices_in_oct(self, xyzs, is_core=False):
        oct = self.oct_core if is_core else self.oct
        map = (xyzs[:, 0] >= oct[1][0]) & (xyzs[:, 0] <= oct[0][0]) \
              & (xyzs[:, 1] >= oct[1][1]) & (xyzs[:, 1] <= oct[0][1]) \
              & (xyzs[:, 2] >= oct[1][2]) & (xyzs[:, 2] <= oct[0][2])

        return np.nonzero(map)[0]

    def split_node(self, oct_overlap_margin, hidden_layers, embedder, is_siren):
        self.haim_net_manager.requires_grad_(False)

        new_radius = self.radius / 2
        top = self.center + new_radius
        btm = self.center - new_radius

        centers = [np.array([top[i] if d == +1 else btm[i] for i, d in enumerate(branch)])
                   for branch in self._directions]

        self.branches = [OctNode(self.csl, center, self, new_radius, oct_overlap_margin, hidden_layers=hidden_layers, embedder=embedder, path=self.path + (direction,), is_siren=is_siren)
                         for center, direction in zip(centers, self._directions)]

        self.is_leaf = False

    def get_mask_for_blending(self, xyzs):
        # return labels for blending in the x direction
        # xyzs are in octant+overlap

        if self.depth == 0:
            # self is root - noting to blend
            return np.full(len(xyzs), 1.0)

        # if not corner-
        # 6 1d interpolation (face)
        # 12 2d interpolation (edge)
        # 8 3d interpolation (vertices)

        # this is not memory efficient
        interpolating_wights = []

        for face_overlap_oct in self._overlapping_octs_around_faces():
            interpolating_wights.append(self._interpolate_oct_wights(face_overlap_oct, xyzs))

        for edge_overlap_oct in self._overlapping_octs_around_edges():
            interpolating_wights.append(self._interpolate_oct_wights(edge_overlap_oct, xyzs))

        for vertex_overlap_oct in self._overlapping_octs_around_vertices():
            interpolating_wights.append(self._interpolate_oct_wights(vertex_overlap_oct, xyzs))

        # this assumes that all interpolation_octs are not intersecting
        wights = np.array([min(ws) for ws in zip(*interpolating_wights)])
        return wights

    def _interpolate_oct_wights(self, interpolation_oct, xyzs):
        interpolation_oct = np.array(interpolation_oct)

        x = np.flip(interpolation_oct[..., 0])
        y = np.flip(interpolation_oct[..., 1])
        z = np.flip(interpolation_oct[..., 2])

        corners = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape((-1, 3))
        corners_in_oct = self.indices_in_oct(corners, is_core=True)

        values = np.full(8, 0.0)
        values[corners_in_oct] = 1.0
        interpolator = RegularGridInterpolator((x, y, z), values.reshape((2, 2, 2)), bounds_error=False, fill_value=1.0)

        assert np.all(interpolator(corners.reshape((-1, 3))) == values)
        return interpolator(xyzs)

    def _overlapping_octs_around_vertices(self):
        if self.depth == 0:
            return []

        overlap_radius = self._overlap_radius
        verts_octs = [[vertex + overlap_radius, vertex - overlap_radius] for vertex, on_boundary in zip(*self._vertices)]

        return np.clip(verts_octs, -1, 1)

    def _overlapping_octs_around_edges(self):
        if self.depth == 0:
            return []

        overlap_radius = self._overlap_radius

        edges_octs = []
        for edge, (is_boundary1, is_boundary2) in zip(*self._edges):

            # an edge is on the boundary iff both of its vertices are on the boundary
            # if not (is_boundary1 and is_boundary2):
            # remove overlapping around vertices
            padding = np.where(edge[0] == edge[1], overlap_radius, -overlap_radius)
            edges_octs.append([np.amax(edge, axis=0) + padding,
                               np.amin(edge, axis=0) - padding])

        # do not to take margin on vertices on boundary
        return np.clip(edges_octs, -1, 1)

    def _overlapping_octs_around_faces(self):
        if self.depth == 0:
            return []

        overlap_radius = self._overlap_radius

        faces_octs = []
        for face, (b1, b2, b3, b4) in zip(*self._faces):
            # a face is on the boundary iff all of its vertices are on the boundary
            # if not (b1 and b2 and b3 and b4):
            # remove overlapping around edges
            padding = np.where((face[0] == face[1]) & (face[0] == face[2]) & (face[0] == face[3]),
                               overlap_radius, -overlap_radius)

            faces_octs.append([np.amax(face, axis=0) + padding,
                               np.amin(face, axis=0) - padding])

        # do not to take margin on vertices on boundary
        return np.clip(faces_octs, -1, 1)


class OctnetTree(INetManager):
    """ Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Cube_with_balanced_ternary_labels.svg/800px-Cube_with_balanced_ternary_labels.svg.png
        """

    def __init__(self, csl, oct_overlap_margin, hidden_layers, embedder, is_siren):
        super().__init__(csl)
        self.csl = csl

        self.root = None
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
        self.oct_overlap_margin = oct_overlap_margin
        self.hidden_layers = hidden_layers
        self.embedder = embedder
        self.depth = None
        self.is_siren = is_siren

    def __str__(self):
        return f'csl={self.csl.model_name} depth={self.depth}'

    def _add_level(self):
        if self.root is None:
            self.root = OctNode(csl=self.csl, center=(0, 0, 0), parent=None, radius=np.array([1, 1, 1]),
                                oct_overlap_margin=self.oct_overlap_margin, hidden_layers=self.hidden_layers,
                                embedder=self.embedder, path=tuple(), is_siren=self.is_siren)
            self.depth = 0
        else:
            [leaf.split_node(self.oct_overlap_margin, self.hidden_layers, self.embedder, self.is_siren) for leaf in self._get_leaves()]
            self.depth += 1

    def _get_leaves(self):
        leaves = []
        self.__get_leaves(self.root, leaves)
        return leaves

    def __get_leaves(self, node, acc):
        if node.is_leaf:
            acc.append(node)
        else:
            [self.__get_leaves(child, acc) for child in node.branches]

    def train_network(self, epochs):
        leaves = self._get_leaves()
        for i, leaf in enumerate(leaves):
            print(f"\nleaf: {i}/{len(leaves) - 1} {leaf.path}")
            leaf.haim_net_manager.train_network(epochs=epochs)

    def prepare_for_training(self, domain_dataset, contour_dataset, hp):
        self._add_level()
        for leaf in self._get_leaves():
            domain_sampler = SubsetRandomSampler(leaf.indices_in_oct(domain_dataset.xyzs))
            contour_sampler = SubsetRandomSampler(leaf.indices_in_oct(contour_dataset.xyzs))

            leaf.haim_net_manager.prepare_for_training(domain_dataset, domain_sampler, contour_dataset, contour_sampler, hp)

    @torch.no_grad()
    def soft_predict(self, xyzs, use_sigmoid=True):
        leaves = self._get_leaves()

        xyzs_per_oct = [xyzs[node.indices_in_oct(xyzs)] for node in leaves]

        labels_per_oct = [node.get_mask_for_blending(xyzs) * node.haim_net_manager.soft_predict(xyzs, use_sigmoid)
                          for node, xyzs in zip(leaves, xyzs_per_oct)]

        return self._merge_per_oct_vals(xyzs, xyzs_per_oct, labels_per_oct)

    def grad_wrt_input(self, xyzs, use_sigmoid=True):
        leaves = self._get_leaves()

        xyzs_per_oct = [xyzs[node.indices_in_oct(xyzs)] for node in leaves]

        # since derivative is a linear operation we can blend them the same way we blend labels
        grad_per_oct = [(node.get_mask_for_blending(xyzs) * node.haim_net_manager.grad_wrt_input(xyzs, use_sigmoid).T).T
                        for node, xyzs in zip(leaves, xyzs_per_oct)]

        return self._merge_per_oct_vals(xyzs, xyzs_per_oct, grad_per_oct)

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        errored_xyzs = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty(0, dtype=bool)

        for leaf in self._get_leaves():
            # ignore interpolation of overlapping
            net_errors_xyzs, net_errors_labels = leaf.haim_net_manager.get_train_errors()

            errored_xyzs = np.concatenate((errored_xyzs, net_errors_xyzs))
            errored_labels = np.concatenate((errored_labels, net_errors_labels))

        return errored_xyzs, errored_labels

    def show_train_losses(self, save_path=None):
        losses_per_leaf = [leaf.haim_net_manager.train_losses for leaf in self._get_leaves()]
        losses = np.mean(losses_per_leaf, axis=0)
        plt.bar(range(len(losses)), losses)
        if save_path is not None:
            plt.savefig(save_path + f"losses_l{self.depth}")
        plt.show()

    @staticmethod
    def _merge_per_oct_vals(xyzs, xyzs_per_oct, values_per_oct):
        flatten_xyzs = (xyz for xyzs in xyzs_per_oct for xyz in xyzs)
        flatten_values = (value for values in values_per_oct for value in values)
        aggregator = {}
        for xyz, value in zip(flatten_xyzs, flatten_values):
            xyz_data = xyz.tobytes()
            if xyz_data in aggregator:
                aggregator[xyz_data] += value
            else:
                aggregator[xyz_data] = value
        return np.array([aggregator[xyz.tobytes()] for xyz in xyzs])
