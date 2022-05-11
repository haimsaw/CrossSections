import multiprocessing

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from Helpers import timing
from Modules import HaimNetWithState
from NetManager import INetManager
from SlicesDataset import SlicesDataset


class ChainTrainer(INetManager):
    def __init__(self, csl, hp, verbose=False):
        super().__init__(csl, verbose)
        self.save_path = "trained_model.pt"
        self.hp = hp

        self.module = HaimNetWithState(hp)
        self.module.double()
        self.module.to(self.device)

        self.bce_loss = None

        self.contour_dataset = None
        self.slices_dataset = None

        self.optimizer = None
        self.lr_scheduler = None
        self.scheduler_step = 0

        self.slices_data_loader = None
        self.contour_data_loader = None

        self.total_epochs = 0
        self.train_losses = []
        self.epochs_with_refine = []
        self.refinements_num = 0

        self.is_training_ready = False

    def _get_constraints(self, slices_xyzs, slices_density, contour_xyzs, normals_on_contour,
                         tangents_on_contour, slice_normals, density_lambda):
        slices_xyzs.requires_grad_(True)
        # contour_xyzs.requires_grad_(True)

        pred_loop = self._forward_loop(slices_xyzs)
        # model_pred_on_contour = self.module(contour_xyzs)

        # grad_on_slices = torch.autograd.grad(model_pred_on_slices.sum(), [slices_xyzs], create_graph=True)[0]
        # grad_on_contour = torch.autograd.grad(model_pred_on_contour.sum(), [contour_xyzs], create_graph=True)[0]

        constraints = {}

        # density - zero inside one outside
        # bce_loss has a sigmoid layer build in
        if self.hp.density_lambda > 0:
            constraints['density'] = sum([self.bce_loss(pred, slices_density) * self.hp.density_lambda for pred in pred_loop])

        return constraints

    def _train_epoch(self, epoch, density_lambda):
        assert self.is_training_ready
        if self.verbose:
            print(f"\n\nEpoch {self.total_epochs} [{epoch}]\n-------------------------------")

        running_loss = 0.0
        running_constraints = {}
        size = len(self.slices_data_loader.dataset)
        is_first = True

        for batch, (slices_xyzs, slices_density) in enumerate(self.slices_data_loader):

            slices_xyzs, slices_density = slices_xyzs.to(self.device), slices_density.to(self.device)
            #contour_xyzs, contour_normals, contour_tangents = contour_xyzs.to(self.device), contour_normals.to(self.device), contour_tangents.to(self.device)
            #slice_normals = slice_normals.to(self.device)

            constraints = self._get_constraints(slices_xyzs, slices_density, None, None,
                                                None, None, density_lambda)
            running_constraints = {k: constraints.get(k, torch.tensor([0])).item() + running_constraints.get(k, 0) for k
                                   in set(constraints)}

            self.optimizer.zero_grad()
            loss = sum(constraints.values())

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if self.verbose and batch % 1000 == 0:
                bce_loss, current = loss.item(), batch * len(slices_xyzs)
                print(f"\tloss: {loss:>7f}, running: {running_loss}  [{current:>5d}/{size:>5d}]")

        if epoch > 0 and epoch % self.scheduler_step == 0:
            self.lr_scheduler.step()

        loss = running_loss / size

        if self.verbose:
            print(f'epoch={epoch} running_loss={running_loss:.2f} losses={running_constraints}')
        self.train_losses.append(loss)

    def _forward_loop(self, xyzs):
        logits = torch.zeros((len(xyzs), 1)).to(self.device)  # todo haim initial values?
        hidden_states = torch.zeros((len(xyzs), self.hp.hidden_state_size)).to(self.device)
        ret = []
        for i in range(self.hp.n_loops):
            logits, hidden_states = self.module(xyzs, logits, hidden_states, torch.tensor([i]).to(self.device))
            ret.append(logits)
        return ret

    def predict_batch(self, xyzs, loop=-1):
        return self._forward_loop(xyzs)[loop]

    def prepare_for_training(self, slices_dataset, contour_dataset):
        # todo haim samplers

        self.slices_dataset = slices_dataset
        self.contour_dataset = contour_dataset

        self.slices_data_loader = DataLoader(self.slices_dataset, batch_size=self.hp.batch_size, shuffle=True)

        self.module.init_weights()

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hp.scheduler_gamma)
        self.scheduler_step = self.hp.scheduler_step

        self.is_training_ready = True

    def update_data_loaders(self, new_cells):
        # todo haim samplers
        print(f'update_data_loaders dataset={len(self.slices_dataset) if self.slices_dataset is not None else 0 }'
              f' new={len(new_cells)}')
        self.slices_dataset = SlicesDataset.from_cells(self.csl, True, new_cells)
        self.slices_data_loader = DataLoader(self.slices_dataset, batch_size=self.hp.batch_size, shuffle=True)
        # contour_sampler = WeightedRandomSampler([1] * len(self.contour_dataset), len(self.slices_dataset) * 2)
        # self.contour_data_loader = DataLoader(self.contour_dataset, batch_size=self.hp.batch_size, sampler=contour_sampler)

    @torch.no_grad()
    def get_refined_cells(self, pool):
        assert self.hp.refinement_type in ['errors', 'edge', 'none']

        self.module.eval()

        # next epoch will be with the refined dataset
        self.epochs_with_refine.append(self.total_epochs + 1)
        size_before = len(self.slices_dataset)
        self.refinements_num += 1

        if self.hp.refinement_type == 'edge':
            xyz_to_refine = self.get_xyz_on_edge()
        elif self.hp.refinement_type == 'errors':
            xyz_to_refine, _ = self.get_train_errors()
        elif self.hp.refinement_type == 'none':
            return
        else:
            raise Exception(f'invalid hp.refinement_type {self.hp.refinement_type}')

        xyz_to_refine = set([x.tobytes() for x in xyz_to_refine])
        new_cells = []
        for cell in self.slices_dataset.cells:
            # todo quadratic - can improve by converting xyz_to_refine to set
            if cell.xyz.tobytes() in xyz_to_refine:
                new_cells += cell.split_cell()
            else:
                new_cells.append(cell)

        promise = pool.map_async(lambda cell: cell.density, new_cells)
        return new_cells, promise


    def train_network(self):


            # todo haim ignore last refine

        if not self.verbose:
            print(f'\ntotal epochs={self.total_epochs}')

    @timing
    def train_epochs_batch(self, epochs):
        self.module.train()

        if not self.verbose:
            print('n_epochs' + '.' * epochs)
            print('_running', end="")
        for epoch in range(epochs):
            if not self.verbose:
                print('.', end='')
            self._train_epoch(epoch, self.hp.density_lambda)
            self.total_epochs += 1
        print('')

    def show_train_losses(self, save_path):
        plt.bar(range(len(self.train_losses)), self.train_losses)
        if save_path is not None:
            plt.savefig(save_path + f"losses")
        # plt.show()

    def load_from_disk(self):
        self.module.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        self.module.eval()

    def save_to_disk(self):
        torch.save(self.module.state_dict(), self.save_path)

    @torch.no_grad()
    def soft_predict(self, xyzs, loop=-1):
        self.module.eval()
        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=float)
        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(self.device)
            batch_labels = self.predict_batch(xyzs_batch, loop)
            label_pred = np.concatenate((label_pred, batch_labels.detach().cpu().numpy().reshape(-1)))
        return label_pred

    def grad_wrt_input(self, xyzs):
        self.module.eval()

        data_loader = DataLoader(xyzs, batch_size=128, shuffle=False)
        grads = np.empty((0, 3), dtype=float)

        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(self.device)
            xyzs_batch.requires_grad_(True)

            self.module.zero_grad()
            pred = self.predict_batch(xyzs_batch)
            grads_batch = torch.autograd.grad(pred.mean(), [xyzs_batch])[0].detach().cpu().numpy()
            grads = np.concatenate((grads, grads_batch))
        return grads

    @torch.no_grad()
    def get_train_errors(self, threshold=0):
        self.module.eval()
        errored_xyzs = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty((0, 1), dtype=bool)

        for xyzs, label in self.slices_data_loader:
            xyzs, label = xyzs.to(self.device), label.to(self.device)

            # xyzs, label_pred = self.hard_predict(xyzs, threshold)
            label_pred = self.predict_batch(xyzs) > threshold  # todo use self.hard_predict (not returning a tensor)

            errors = (label != label_pred).view(-1)

            errored_xyzs = np.concatenate((errored_xyzs, xyzs[errors].detach().cpu().numpy()))
            errored_labels = np.concatenate((errored_labels, label[errors].detach().cpu().numpy()))

        return errored_xyzs, errored_labels.reshape(-1)

    @torch.no_grad()
    def get_xyz_on_edge(self):
        self.module.eval()
        xyzs_at_edge = np.empty((0, 3), dtype=np.double)

        for xyzs, labels in self.slices_data_loader:
            xyzs, labels = xyzs.to(self.device), labels.to(self.device)

            xyzs_at_edge = np.concatenate((xyzs_at_edge, xyzs[torch.logical_and(0 < labels, labels < 1).view(-1)]
                                           .detach().cpu().numpy()))

        return xyzs_at_edge
