import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn

from Modules import *
from Helpers import *
from abc import ABCMeta, abstractmethod
from torch import linalg as LA


class INetManager:
    __metaclass__ = ABCMeta

    def __init__(self, csl, verbose=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.csl = csl
        self.verbose = verbose

    @abstractmethod
    def show_train_losses(self, save_path): raise NotImplementedError

    @abstractmethod
    def load_from_disk(self): raise NotImplementedError

    @abstractmethod
    def save_to_disk(self): raise NotImplementedError

    @abstractmethod
    def requires_grad_(self, requires_grad): raise NotImplementedError

    @abstractmethod
    def prepare_for_training(self, dataset, lr, scheduler_step, weight_decay, eikonal_lambda, sampler): raise NotImplementedError

    @abstractmethod
    def train_network(self, epochs): raise NotImplementedError

    @abstractmethod
    def get_train_errors(self, threshold=0.5): raise NotImplementedError

    @abstractmethod
    def soft_predict(self, xyzs, use_sigmoid=True): raise NotImplementedError

    @abstractmethod
    def grad_wrt_input(self, xyzs, use_sigmoid=True): raise NotImplementedError

    @torch.no_grad()
    def hard_predict(self, xyzs, threshold=0.5):
        # todo self.module.eval()
        soft_labels = self.soft_predict(xyzs)
        return soft_labels > threshold


class HaimNetManager(INetManager):
    def __init__(self, csl, hidden_layers, embedder, residual_module, octant, is_siren, verbose=False):
        super().__init__(csl, verbose)

        self.save_path = "trained_model.pt"
        self.octant = octant

        self.module = HaimNet(hidden_layers, residual_module, embedder, is_siren)
        self.module.double()
        self.module.to(self.device)

        self.bce_loss = None
        self.eikonal_lambda = None

        self.optimizer = None
        self.lr_scheduler = None
        self.scheduler_step = 0

        self.data_loader = None

        self.total_epochs = 0
        self.train_losses = []

        self.is_training_ready = False

    def _train_epoch(self, epoch):
        assert self.is_training_ready
        if self.verbose:
            print(f"\n\nEpoch {self.total_epochs} [{epoch}]\n-------------------------------")

        running_loss = 0.0
        size = len(self.data_loader.dataset)
        for batch, (xyz, label) in enumerate(self.data_loader):
            xyz, label = xyz.to(self.device), label.to(self.device)
            xyz.requires_grad_(True)
            self.optimizer.zero_grad()

            # Compute prediction error
            label_pred = self.module(xyz)

            # BCEWithLogitsLoss contains sigmoid
            # todo haim should I use sigmoid for calculating eikonal and boundary loss?
            bce_loss = self.bce_loss(label_pred, label)
            bce_loss.backward(create_graph=True, retain_graph=True)

            # eikonal
            if self.eikonal_lambda > 0:
                eikonal_loss = self.eikonal_lambda * torch.mean(torch.abs(LA.vector_norm(xyz.grad, dim=-1) - 1))
                eikonal_loss.backward()

            self.optimizer.step()

            running_loss += bce_loss.item() * len(xyz)
            if self.verbose and batch % 1000 == 0:
                bce_loss, current = bce_loss.item(), batch * len(xyz)
                print(f"\tloss: {bce_loss:>7f}, running: {running_loss}  [{current:>5d}/{size:>5d}]")

        if epoch > 0 and epoch % self.scheduler_step == 0:
            self.lr_scheduler.step()

        total_loss = running_loss / size
        if self.verbose:
            print(f"\tloss for epoch: {total_loss}")
        self.train_losses.append(total_loss)

    def prepare_for_training(self, dataset, lr, scheduler_step, weight_decay, eikonal_lambda, sampler):
        self.data_loader = DataLoader(dataset, batch_size=128, sampler=sampler)

        self.module.init_weights()

        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.eikonal_lambda = eikonal_lambda

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler_step = scheduler_step

        self.is_training_ready = True

    def train_network(self, epochs):
        if not self.verbose:
            print('n_epochs' + '.' * epochs)
            print('_running', end="")

        self.module.train()
        for epoch in range(epochs):
            if not self.verbose:
                print('.', end='')
            self._train_epoch(epoch)
            self.total_epochs += 1

        if not self.verbose:
            print(f'\ntotal epochs={self.total_epochs}')

    def show_train_losses(self, save_path):
        plt.bar(range(len(self.train_losses)), self.train_losses)
        if save_path is not None:
            plt.savefig(save_path + f"losses")
        plt.show()

    def load_from_disk(self):
        self.module.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        self.module.eval()

    def save_to_disk(self):
        torch.save(self.module.state_dict(), self.save_path)

    def requires_grad_(self, requires_grad):
        self.module.requires_grad_(requires_grad)

    @torch.no_grad()
    def soft_predict(self, xyzs, use_sigmoid=True):
        # todo assert in octant

        self.module.eval()
        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=float)
        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(self.device)
            batch_labels = torch.sigmoid(self.module(xyzs_batch)) if use_sigmoid else self.module(xyzs_batch)
            label_pred = np.concatenate((label_pred, batch_labels.detach().cpu().numpy().reshape(-1)))
        return label_pred

    def grad_wrt_input(self, xyzs, use_sigmoid=True):
        # todo assert in octant

        self.module.eval()

        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        grads = np.empty((0, 3), dtype=float)

        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(self.device)
            xyzs_batch.requires_grad_(True)

            self.module.zero_grad()
            (torch.sigmoid(self.module(xyzs_batch)) if use_sigmoid else self.module(xyzs_batch)).mean().backward()

            grads_batch = xyzs_batch.grad.detach().cpu().numpy()
            grads = np.concatenate((grads, grads_batch))
        return grads

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        self.module.eval()
        errored_xyzs = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty((0, 1), dtype=bool)

        for xyzs, label in self.data_loader:
            xyzs, label = xyzs.to(self.device), label.to(self.device)

            # xyzs, label_pred = self.hard_predict(xyzs, threshold)
            label_pred = self.module(xyzs) > threshold  # todo use self.hard_predict (not returning a tensor)

            errors = (label != label_pred).view(-1)

            errored_xyzs = np.concatenate((errored_xyzs, xyzs[errors].detach().cpu().numpy()))
            errored_labels = np.concatenate((errored_labels, label[errors].detach().cpu().numpy()))

        return errored_xyzs, errored_labels.reshape(-1)
