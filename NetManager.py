import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from Modules import HaimNet
from Resterizer import rasterizer_factory
from Helpers import *
from abc import ABCMeta, abstractmethod


class RasterizedCslDataset(Dataset):
    def __init__(self, csl, sampling_resolution=(256, 256), sampling_margin=0.2, octant=None, transform=None, target_transform=None):
        self.csl = csl

        cells = []
        for plane in csl.planes:
            cells += rasterizer_factory(plane).get_rasterazation_cells(sampling_resolution, sampling_margin, octant)

        self.cells = np.array(cells)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.cells.size

    def __getitem__(self, idx):
        cell = self.cells[idx]

        xyz = cell.xyz
        label = [cell.label]

        if self.transform:
            xyz = self.transform(xyz)
        if self.target_transform:
            label = self.target_transform(label)

        return xyz, label

    # todo - not correct anymore, refined cells might end up in wrong octant
    def refine_cells(self, xyz_to_refine):
        # xyz_to_refine = set(xyz_to_refine)

        new_cells = []
        for cell in self.cells:
            # todo quadratic - can improve by converting xyz_to_refine to set
            if cell.xyz in xyz_to_refine:
                new_cells += cell.split_cell()
            else:
                new_cells.append(cell)

        self.cells = np.array(new_cells)


class INetManager:
    __metaclass__ = ABCMeta

    @abstractmethod
    def soft_predict(self, xyzs): raise NotImplementedError

    @abstractmethod
    def prepare_for_training(self, sampling_resolution_2d, sampling_margin, lr): raise NotImplementedError

    @abstractmethod
    def train_network(self, epochs): raise NotImplementedError


class HaimNetManager(INetManager):
    def __init__(self, csl, layers, residual_module=None, octant=None, verbose=False):
        self.verbose = verbose
        self.save_path = "trained_model.pt"
        self.octant = octant
        self.csl = csl

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.module = HaimNet(layers, residual_module).to(self.device)
        self.module.double()

        print(self.module)

        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None

        self.data_loader = None
        self.dataset = None

        self.total_epochs = 0
        self.epochs_with_refine = []
        self.train_losses = []
        self.refinements_num = 0

        self.is_training_ready = False

    def _train_epoch(self, epoch):
        assert self.is_training_ready
        if self.verbose:
            print(f"\n\nEpoch {self.total_epochs} [{epoch}]\n-------------------------------")

        running_loss = 0.0
        size = len(self.data_loader.dataset)
        for batch, (xyz, label) in enumerate(self.data_loader):
            xyz, label = xyz.to(self.device), label.to(self.device)

            # Compute prediction error
            label_pred = self.module(xyz)
            # print(f"{label_pred.shape}, {label.shape}")
            loss = self.loss_fn(label_pred, label)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * len(xyz)
            if self.verbose and batch % 1000 == 0:
                loss, current = loss.item(), batch * len(xyz)
                print(f"\tloss: {loss:>7f}, running: {running_loss}  [{current:>5d}/{size:>5d}]")

        self.lr_scheduler.step()

        total_loss = running_loss/size
        if self.verbose:
            print(f"\tloss for epoch: {total_loss}")
        self.train_losses.append(total_loss)

    def prepare_for_training(self, sampling_resolution, sampling_margin, lr):
        self.dataset = RasterizedCslDataset(self.csl, sampling_resolution=sampling_resolution, sampling_margin=sampling_margin,
                                            octant=self.octant, target_transform=torch.tensor, transform=torch.tensor)
        self.data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)

        self.module.init_weights()

        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.is_training_ready = True

    def train_network(self, epochs):
        if not self.verbose:
            print('\nn_epochs' + '.' * epochs)
            print('_running', end="")

        self.module.train()
        for epoch in range(epochs):
            if not self.verbose:
                print('.', end='')
            self._train_epoch(epoch)
            self.total_epochs += 1

        if not self.verbose:
            print(f'\ntotal epochs={self.total_epochs}')

    def show_train_losses(self):
        colors = ['red' if i in self.epochs_with_refine else 'blue' for i in range(len(self.train_losses))]
        plt.bar(range(len(self.train_losses)), self.train_losses, color=colors)
        plt.show()

    def load_from_disk(self):
        self.module.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        self.module.eval()

    def save_to_disk(self):
        torch.save(self.module.state_dict(), self.save_path)

    def requires_grad_(self, requires_grad):
        self.module.requires_grad_(requires_grad)

    @torch.no_grad()
    def hard_predict(self, xyzs, threshold=0.5):
        self.module.eval()
        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=bool)
        for xyz_batch in data_loader:
            xyz_batch = xyz_batch.to(self.device)
            label_pred = np.concatenate((label_pred, (self.module(xyz_batch) > threshold).detach().cpu().numpy().reshape(-1)))
        return label_pred

    @torch.no_grad()
    def soft_predict(self, xyzs):
        self.module.eval()
        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=float)
        for xyz_batch in data_loader:
            xyz_batch = xyz_batch.to(self.device)
            label_pred = np.concatenate((label_pred, self.module(xyz_batch).detach().cpu().numpy().reshape(-1)))
        return label_pred

    @torch.no_grad()
    def refine_sampling(self, threshold=0.5):
        self.module.eval()

        # next epoch will be with the refined dataset
        self.epochs_with_refine.append(self.total_epochs + 1)
        size_before = len(self.dataset)
        self.refinements_num += 1

        errored_xyz, _ = self.get_train_errors()

        self.dataset.refine_cells(errored_xyz)
        print(f'refine_sampling before={size_before}, after={len(self.dataset)}, n_refinements = {self.refinements_num}')

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        self.module.eval()
        errored_xyz = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty((0, 1), dtype=bool)

        for xyz, label in self.data_loader:
            xyz, label = xyz.to(self.device), label.to(self.device)

            label_pred = self.module(xyz) > threshold
            errors = (label != label_pred).view(-1)

            errored_xyz = np.concatenate((errored_xyz, xyz[errors].detach().cpu().numpy()))
            errored_labels = np.concatenate((errored_labels, label[errors].detach().cpu().numpy()))

        return errored_xyz, errored_labels.reshape(-1)


class OctnetreeManager(INetManager):
    def __init__(self, csl, layers, network_manager_root, sampling_margin):
        self.octanes = get_octets(*add_margin(*get_top_bottom(csl.all_vertices), sampling_margin))
        self.network_managers = [HaimNetManager(csl, layers, residual_module=network_manager_root.module, octant=octant)
                                 for octant in self.octanes]

    def prepare_for_training(self, sampling_resolution_2d, sampling_margin, lr):
        for network_manager in self.network_managers:
            network_manager.prepare_for_training(sampling_resolution_2d, sampling_margin, lr)

    def train_network(self, epochs):
        for network_manager in self.network_managers:
            network_manager.train_network(epochs=epochs)
            # network_manager.show_train_losses()

    @torch.no_grad()
    def soft_predict(self, xyzs):
        # todo do in GPU
        return np.array([self.predict_xyz(xyz) for xyz in xyzs])

    @torch.no_grad()
    def predict_xyz(self, xyz):
        for network_manager, octant in zip(self.network_managers, self.octanes):
            if is_in_octant(xyz, octant):
                return network_manager.soft_predict([xyz])
        raise Exception(f"xyz not in any octant. xyz={xyz}")
