import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from NetManager import *


class OctNode:
    def __init__(self, csl, center, parent, radius, oct_overlap_margin, path, **haimnet_kwargs):
        # top bottom = center +- radius
        self.center = center

        self.csl = csl
        self.radius = radius
        self.oct_overlap_margin = oct_overlap_margin
        self.branches = None
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
        self.is_leaf = True
        self.path = path

        res = None if parent is None else parent.haim_net_manager.module
        self.haim_net_manager = HaimNetManager(csl, residual_module=res, octant=self.oct, **haimnet_kwargs)

    def __str__(self):
        return f"position: {self.center}, radius: {self.radius}"

    @property
    def oct(self):
        radius_with_margin = self.radius * (1 + 2 * self.oct_overlap_margin)
        return np.stack((self.center + radius_with_margin, self.center - radius_with_margin))

    @property
    def oct_core(self):
        return np.stack((self.center + self.radius, self.center - self.radius))

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

    def indices_in_oct(self, xyzs, is_core=False):
        oct = self.oct_core if is_core else self.oct
        map = (xyzs[:, 0] >= oct[1][0]) & (xyzs[:, 0] <= oct[0][0]) \
              & (xyzs[:, 1] >= oct[1][1]) & (xyzs[:, 1] <= oct[0][1]) \
              & (xyzs[:, 2] >= oct[1][2]) & (xyzs[:, 2] <= oct[0][2])

        return np.nonzero(map)[0]

    def split_node(self, oct_overlap_margin, hidden_layers, embedder):
        new_radius = self.radius/2
        top = self.center + new_radius
        btm = self.center - new_radius

        centers = [np.array([top[i] if d == '+' else btm[i] for i, d in enumerate(branch)])
                   for branch in self.branches_directions]

        self.branches = [OctNode(self.csl, center, self, new_radius, oct_overlap_margin, hidden_layers=hidden_layers, embedder=embedder, path=self.path + (direction,))
                         for center, direction in zip(centers, self.branches_directions)]

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
            lines.append( line_getter_neg(i))
            lines.append(line_getter_pos(i))

        # wights = np.array([min(1, *[l(xyz) for l in lines]) for xyz in xyzs])
        wights = np.full(len(xyzs), 1.0)
        # wights = np.full(len(xyzs), 0.0)[self.indices_in_oct(xyzs, is_core = True)] = 1.0
        return wights

    def get_mask_for_blending(self, xyzs):
        # return labels for blending in the x direction
        # xyzs are in octant+overlap
        # todo this assumes that octree depth is 1

        if len(self.path) == 0:
            # self is root - noting to blend
            return np.full(len(xyzs), 1.0)
        # todo remove this
        #if self.path[-1] != '---':
            #return np.full(len(xyzs), 0.0)

        # if not corner-
        # 6 1d interpolation (face)
        # 12 2d interpolation (edge)
        # 8 3d interpolation (vertices)

        core_start = self.oct_core[1]
        core_end = self.oct_core[0]

        margin_start = self.oct[1]
        margin_end = self.oct[0]

        non_blending_start = 2 * core_start - margin_start
        non_blending_end = 2 * core_end - margin_end

        x = np.linspace(-0.2, 0.2, 2)
        y = np.linspace(-0.2, 0.2, 2)
        z = np.linspace(-0.2, 0.2, 2)
        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        # data = f(xg, yg, zg)

        # 3d interpolation
        points = (x, y, z)  # all points on the cube
        values = np.full((2, 2, 2), 0.0)
        idx = tuple(0 if d == '-' else 1 for d in self.path[0])
        values[idx] = 1.0

        my_interpolating_function = RegularGridInterpolator(points, values, bounds_error=False, fill_value=1.0)
        wights = my_interpolating_function(xyzs)
        return wights


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
        labels_per_oct = [node.get_mask_for_blending(xyzs)  # * node.haim_net_manager.soft_predict(xyzs, use_sigmoid) # todo this
                          for node, xyzs in zip(leaves, xyzs_per_oct)]

        return self._merge_oct_predictions(xyzs, labels_per_oct, xyzs_per_oct)

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        errored_xyzs = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty(0, dtype=bool)

        for leaf in self._get_leaves():
            # todo handle ovelapping?
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
        dict = {}
        for xyz, label in zip(flatten_xyzs, flatten_labels):
            xyz_data = xyz.tobytes()
            if xyz_data in dict:
                dict[xyz_data] += label
            else:
                dict[xyz_data] = label
        labels = np.array([dict[xyz.tobytes()] for xyz in xyzs])
        return labels


