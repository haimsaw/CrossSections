import numpy as np

from NetManager import *


class OctNode:
    def __init__(self, csl, center, parent, radius, overlap_margin, **haimnet_kwargs):
        # top botom = center +- radius
        self.center = center
        self.haimNetManager = HaimNetManager(csl, residual_module=parent, **haimnet_kwargs)
        self.csl = csl
        self.radius = radius
        self.overlap_margin = overlap_margin
        self.branches = None
        self.parent = parent
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
        self.haimnet_kwargs = haimnet_kwargs
        self.is_leaf = True

    def train_node(self, root_sampling_resolution_2d, sampling_margin, lr, scheduler_step, n_epochs):
        self.haimNetManager.prepare_for_training(root_sampling_resolution_2d, sampling_margin, lr, scheduler_step)
        self.haimNetManager.train_network(epochs=n_epochs)

    def split_node(self):
        new_radius = self.radius/2
        top = self.center + new_radius
        btm = self.center - new_radius

        centers = [np.array([top[i] if d == '+' else btm[i] for i, d in enumerate(branch)])
                   for branch in self.branches_directions]

        self.branches = [OctNode(self.csl, center, self, new_radius, self.overlap_margin, **self.haimnet_kwargs)
                         for center in centers]

        self.is_leaf = False


    @staticmethod
    def __findBranch(root, position):
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


class OctnetTree:
    """ Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Cube_with_balanced_ternary_labels.svg/800px-Cube_with_balanced_ternary_labels.svg.png
        """
    def __init__(self, csl, overlap_margin, hidden_layers, embedder):
        self.csl = csl
        self.root = OctNode(csl=csl, center=(0, 0, 0), parent=None, radius=(1, 1, 1), overlap_margin=overlap_margin, hidden_layers=hidden_layers, embedder=embedder)
        self.branches_directions = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")

    def train_leaves(self, **train_kwargs):
        return self.train_leaves(node=self.root, **train_kwargs)

    def add_level(self):
        return self._add_level(node=self.root)

    def _add_level(self, node):

        if node.is_leaf:
            node.split_node()
        else:
            [self._add_level(child) for child in node.branches]

    def _train_leaves(self, node, **train_kwargs):
        if node.is_leaf:
            node.train_node(**train_kwargs)
        else:
            [self._train_leaves(child) for child in node.branches]



