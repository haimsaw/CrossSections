import numpy as np
import torch

from NetManager import *


class OctNode:
    def __init__(self, csl, center, parent, radius, overlap_margin, **haimnet_kwargs):
        # top bottom = center +- radius
        self.center = center

        self.csl = csl
        self.radius = radius
        self.overlap_margin = overlap_margin
        self.branches = None
        self.parent = parent
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
        self.haim_net_kwargs = haimnet_kwargs
        self.is_leaf = True

        # todo make this private
        res = None if parent is None else parent.haim_net_manager.module
        self.haim_net_manager = HaimNetManager(csl, residual_module=res, octant=self.oct, **haimnet_kwargs)

    def __str__(self):
        return f"position: {self.center}, radius: {self.radius}"

    @property
    def oct(self):
        radius_with_margin = self.radius * (1 + self.overlap_margin)
        return np.stack((self.center + radius_with_margin, self.center - radius_with_margin))

    @property
    def oct_core(self):
        return np.stack((self.center + self.radius, self.center - self.radius))

    def train_leaf(self, *, dataset, lr, scheduler_step, n_epochs):
        assert self.is_leaf

        self.haim_net_manager.prepare_for_training(dataset, lr, scheduler_step)
        self.haim_net_manager.train_network(epochs=n_epochs)

    def split_node(self):
        new_radius = self.radius/2
        top = self.center + new_radius
        btm = self.center - new_radius

        centers = [np.array([top[i] if d == '+' else btm[i] for i, d in enumerate(branch)])
                   for branch in self.branches_directions]

        self.branches = [OctNode(self.csl, center, self, new_radius, self.overlap_margin, **self.haim_net_kwargs)
                         for center in centers]

        self.is_leaf = False
        self.haim_net_manager.requires_grad_(False)

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


class OctnetTree(INetManager):
    """ Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Cube_with_balanced_ternary_labels.svg/800px-Cube_with_balanced_ternary_labels.svg.png
        """
    def __init__(self, csl, overlap_margin, hidden_layers, embedder):
        super().__init__(csl)
        self.csl = csl
        self.root = OctNode(csl=csl, center=(0, 0, 0), parent=None, radius=np.array([1, 1, 1]), overlap_margin=overlap_margin, hidden_layers=hidden_layers, embedder=embedder)
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")

    def train_leaves(self, sampling_resolution, sampling_margin, **train_kwargs):
        # todo save these in a list of levels (for get error)?
        # todo extract this to OctnetTree.prepere for training
        dataset = RasterizedCslDataset(self.csl, sampling_resolution=sampling_resolution, sampling_margin=sampling_margin,
                                       target_transform=torch.tensor, transform=torch.tensor)

        leaves = self._get_leaves()
        for i, leaf in enumerate(leaves):
            print(f"\nleaf: {i}/{len(leaves) - 1} ")
            leaf.train_leaf(dataset=dataset, **train_kwargs)

    def add_level(self):
        [leaf.split_node() for leaf in self._get_leaves()]

    def _get_leaves(self):
        leaves = []
        self.__get_leaves(self.root, leaves)
        return leaves

    def __get_leaves(self, node, acc):
        if node.is_leaf:
            acc.append(node)
        else:
            [self.__get_leaves(child, acc) for child in node.branches]

    @torch.no_grad()
    def soft_predict(self, xyzs, use_sigmoid=True):
        leaves = self._get_leaves()

        xyzs_per_oct = [xyzs[is_in_octant_list(xyzs, node.oct)] for node in leaves]
        labels_per_oct = [get_mask_for_blending_old(xyzs, node.oct, node.oct_core, direction) * node.haim_net_manager.soft_predict(xyzs, use_sigmoid)
                          for node, xyzs, direction in zip(leaves, xyzs_per_oct, self.branches_directions)]

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

    def _merge_oct_predictions(self, xyzs, labels_per_oct, xyzs_per_oct):
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


