import numpy as np
from scipy.interpolate import RegularGridInterpolator
from NetManager import *
from itertools import combinations


class OctNode:
    def __init__(self, csl, center, parent, radius, oct_overlap_margin, path, **haimnet_kwargs):
        # top bottom = center +- radius
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

    def indices_in_oct(self, xyzs, is_core=False):
        oct = self.oct_core if is_core else self.oct
        map = (xyzs[:, 0] >= oct[1][0]) & (xyzs[:, 0] <= oct[0][0]) \
              & (xyzs[:, 1] >= oct[1][1]) & (xyzs[:, 1] <= oct[0][1]) \
              & (xyzs[:, 2] >= oct[1][2]) & (xyzs[:, 2] <= oct[0][2])

        return np.nonzero(map)[0]

    def split_node(self, oct_overlap_margin, hidden_layers, embedder):
        new_radius = self.radius / 2
        top = self.center + new_radius
        btm = self.center - new_radius

        centers = [np.array([top[i] if d == +1 else btm[i] for i, d in enumerate(branch)])
                   for branch in self._directions]

        self.branches = [OctNode(self.csl, center, self, new_radius, oct_overlap_margin, hidden_layers=hidden_layers, embedder=embedder, path=self.path + (direction,))
                         for center, direction in zip(centers, self._directions)]

        self.is_leaf = False
        self.haim_net_manager.requires_grad_(False)

    def get_mask_for_blending_old(self, xyzs):
        # return labels for blending in the x direction
        # xyzs are in octant+overlap

        core_end = self.oct_core[0]
        core_start = self.oct_core[1]
        margin_end = self.oct[0]
        margin_start = self.oct[1]

        non_blending_start = 2 * core_start - margin_start
        non_blending_end = 2 * core_end - margin_end

        line_getter_pos = lambda i: lambda xyz: xyz[i] * 1 / (non_blending_start[i] - margin_start[i]) + margin_start[i] / (margin_start[i] - non_blending_start[i])
        line_getter_neg = lambda i: lambda xyz: xyz[i] * 1 / (non_blending_end[i] - margin_end[i]) + margin_end[i] / (margin_end[i] - non_blending_end[i])

        lines = []
        for i in range(3):
            lines.append(line_getter_neg(i))
            lines.append(line_getter_pos(i))

        wights = np.array([min(1, *[l(xyz) for l in lines]) for xyz in xyzs])
        # wights = np.full(len(xyzs), 1.0)
        # wights = np.full(len(xyzs), 0.0)[self.indices_in_oct(xyzs, is_core = True)] = 1.0
        return wights

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

        '''core_start = self.oct_core[1]
        core_end = self.oct_core[0]

        margin_start = self.oct[1]
        margin_end = self.oct[0]

        non_blending_start = 2 * core_start - margin_start
        non_blending_end = 2 * core_end - margin_end
        '''

        # vertices interpolation
        interpolating_wights = []
        for vertex_overlap_oct in self._overlapping_octs_around_vertices():
            interpolating_wights.append(self._get_interpolation_wights(vertex_overlap_oct, xyzs))

        # edges interpolation
        for edge_overlap_oct in self._overlapping_octs_around_edges():
            interpolating_wights.append(self._get_interpolation_wights(edge_overlap_oct, xyzs))

        # this assumes that all interpolation_octs are not interesting
        wights = [min(ws) for ws in zip(*interpolating_wights)]
        return wights

    def _get_interpolation_wights(self, interpolation_oct, xyzs):
        x = np.linspace(interpolation_oct[1][0], interpolation_oct[0][0], 2)
        y = np.linspace(interpolation_oct[1][1], interpolation_oct[0][1], 2)
        z = np.linspace(interpolation_oct[1][2], interpolation_oct[0][2], 2)
        points = (x, y, z)
        corners_in_oct = self.indices_in_oct(np.stack(np.meshgrid(x, y, z), axis=-1).reshape((-1, 3)), is_core=True)
        values = np.full(8, 0.0)
        values[corners_in_oct] = 1.0
        values = values.reshape((2, 2, 2))
        my_interpolating_function = RegularGridInterpolator(points, values, bounds_error=False, fill_value=1.0)
        return my_interpolating_function(xyzs)

    def _overlapping_octs_around_vertices(self):
        if self.depth == 0:
            return []

        # todo property 2*self.radius*overlapping
        overlap_radius = np.array([0.2 if self.depth == 1 else 0.05] * 3)

        return [[vertex + overlap_radius, vertex - overlap_radius]
                for vertex, on_boundary in zip(self._vertices) if not on_boundary]

    def _overlapping_octs_around_edges(self):
        if self.depth == 0:
            return []
        vertices, vertices_is_on_boundary = self._vertices
        edges_directions = self._directions

        legal_edges = [sum(d1 * d2) == 1 for d1, d2 in combinations((edges_directions, edges_directions), 2)]

        edges = combinations((vertices, vertices), 2)

        # an edge is on the boundary iff both of its vertices are on the boundary
        edges_on_boundary = [b1 and b2 for b1, b2 in combinations((vertices_is_on_boundary, vertices_is_on_boundary), 2)]

        # consider only edges on from adjacent vertices (no diagonals)
        # edge is not a diagonals if its vertices agree in two coordinates

        return None


class OctnetTree(INetManager):
    """ Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Cube_with_balanced_ternary_labels.svg/800px-Cube_with_balanced_ternary_labels.svg.png
        """

    def __init__(self, csl, oct_overlap_margin, hidden_layers, embedder):
        super().__init__(csl)
        self.csl = csl

        self.root = None
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
        self.oct_overlap_margin = oct_overlap_margin
        self.hidden_layers = hidden_layers
        self.embedder = embedder

    def _add_level(self):
        if self.root is None:
            self.root = OctNode(csl=self.csl, center=(0, 0, 0), parent=None, radius=np.array([1, 1, 1]), oct_overlap_margin=self.oct_overlap_margin, hidden_layers=self.hidden_layers, embedder=self.embedder, path=tuple())
        else:
            [leaf.split_node(self.oct_overlap_margin, self.hidden_layers, self.embedder) for leaf in self._get_leaves()]

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
            print(f"\nleaf: {i}/{len(leaves) - 1} ")
            leaf.haim_net_manager.train_network(epochs=epochs)

    def prepare_for_training(self, dataset, lr, scheduler_step, sampler=None):
        self._add_level()
        for leaf in self._get_leaves():
            sampler = SubsetRandomSampler(leaf.indices_in_oct(dataset.xyzs))
            leaf.haim_net_manager.prepare_for_training(dataset, lr, scheduler_step, sampler)

    @torch.no_grad()
    def soft_predict(self, xyzs, use_sigmoid=True):
        leaves = self._get_leaves()

        xyzs_per_oct = [xyzs[node.indices_in_oct(xyzs)] for node in leaves]
        labels_per_oct = [node.get_mask_for_blending(xyzs)  # * node.haim_net_manager.soft_predict(xyzs, use_sigmoid) # todo HAIM this
                          for node, xyzs in zip(leaves, xyzs_per_oct)]

        return self._merge_oct_predictions(xyzs, labels_per_oct, xyzs_per_oct)

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        errored_xyzs = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty(0, dtype=bool)

        for leaf in self._get_leaves():
            # todo handle overlapping?
            net_errors_xyzs, net_errors_labels = leaf.haim_net_manager.get_train_errors()

            errored_xyzs = np.concatenate((errored_xyzs, net_errors_xyzs))
            errored_labels = np.concatenate((errored_labels, net_errors_labels))

        return errored_xyzs, errored_labels

    def show_train_losses(self):
        # todo - aggregate all losses
        for i, leaf in enumerate(self._get_leaves()):
            print(f"leaf: {i}")
            leaf.haim_net_manager.show_train_losses()

    @staticmethod
    def _merge_oct_predictions(xyzs, labels_per_oct, xyzs_per_oct):
        flatten_xyzs = (xyz for xyzs in xyzs_per_oct for xyz in xyzs)
        flatten_labels = (label for labels in labels_per_oct for label in labels)
        aggregator = {}
        for xyz, label in zip(flatten_xyzs, flatten_labels):
            xyz_data = xyz.tobytes()
            if xyz_data in aggregator:
                aggregator[xyz_data] += label
            else:
                aggregator[xyz_data] = label
        labels = np.array([aggregator[xyz.tobytes()] for xyz in xyzs])
        return labels
