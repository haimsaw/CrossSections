import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from Resterizer import rasterizer_factory


class RasterizedCslDataset(Dataset):
    def __init__(self, csl, sampling_resolution=(256, 256), sampling_margin=0.2, transform=None, target_transform=None):
        self.csl = csl

        self.cells = np.array([rasterizer_factory(plane).get_rasterazation_cells(sampling_resolution, sampling_margin)
                               for plane in csl.planes]).reshape(-1)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.cells.size

    def __getitem__(self, idx):
        cell = self.cells[idx]

        xyz = cell.xyz
        label = [1.0] if cell.label else [0.0]

        if self.transform:
            xyz = self.transform(xyz)
        if self.target_transform:
            label = self.target_transform(label)

        return xyz, label

    def refine_cells(self, xyz_to_refine):

        xyz_to_refine = set(xyz_to_refine)
        new_cells = []
        for cell in self.cells:
            if cell.xyz in xyz_to_refine:
                new_cells += cell.split_cell()
            else:
                new_cells.append(cell)

        self.cells = np.array(new_cells)


class NaiveNetwork(nn.Module):
    def __init__(self):
        super(NaiveNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(3, 128),   nn.LeakyReLU(),
            nn.Linear(128, 256), nn.LeakyReLU(),
            nn.Linear(256, 512), nn.LeakyReLU(),
            nn.Linear(512, 512), nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def init_weights(self):
        def initializer(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.constant_(m.weight, 1)
                m.bias.data.fill_(0.1)
        self.linear_relu.apply(initializer)

    def forward(self, x):
        return self.linear_relu(x)


class NetworkManager:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.save_path = "trained_model.pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.model = NaiveNetwork().to(self.device)
        self.model.double()

        print(self.model)

        self.loss_fn = None
        self.train_losses = []
        self.optimizer = None
        self.data_loader = None
        self.dataset = None

        self.is_training_ready = False
        self.total_epochs = 0
        self.epochs_with_refine = []

    def _train_epoch(self, epoch):
        assert self.is_training_ready
        if self.verbose:
            print(f"\n\nEpoch {self.total_epochs} [{epoch}]\n-------------------------------")

        running_loss = 0.0
        size = len(self.data_loader.dataset)
        for batch, (xyz, label) in enumerate(self.data_loader):
            xyz, label = xyz.to(self.device), label.to(self.device)

            # Compute prediction error
            label_pred = self.model(xyz)
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
        total_loss = running_loss/size
        if self.verbose:
            print(f"\tloss for epoch: {total_loss}")
        self.train_losses.append(total_loss)

    def prepare_for_training(self, csl, sampling_resolution, sampling_margin, lr):
        self.dataset = RasterizedCslDataset(csl, sampling_resolution=sampling_resolution, sampling_margin=sampling_margin,
                                            target_transform=torch.tensor, transform=torch.tensor)
        self.data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        self.model.init_weights()
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.is_training_ready = True

        for xyz, label in self.data_loader:
            print("Shape of X [N, C, H, W]: ", xyz.shape)
            print("Shape of label: ", label.shape, label.dtype)
            break

    def train_network(self, epochs):
        if not self.verbose:
            print('\nn_epochs' + '.' * epochs)
            print('_running', end="")

        self.model.train()
        for epoch in range(epochs):
            if not self.verbose:
                print('.', end='')
            self._train_epoch(epoch)
            self.total_epochs += 1

        if not self.verbose:
            print(f'\ntotal epochs={self.total_epochs}')

    def show_train_losses(self):
        colors = ['red' if i in self.epochs_with_refine else 'blue' for i in range(len(self.train_losses))]
        plt.bar(range(len(self.train_losses)), self.train_losses, color=colors, figsize=(10,10))
        plt.show()

    def load_from_disk(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        self.model.eval()

    def save_to_disk(self):
        torch.save(self.model.state_dict(), self.save_path)

    @torch.no_grad()
    def hard_predict(self, xyz, threshold=0.5):
        self.model.eval()
        data_loader = DataLoader(xyz, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=bool)
        for xyz_batch in data_loader:
            xyz_batch = xyz_batch.to(self.device)
            label_pred = np.concatenate((label_pred, (self.model(xyz_batch) > threshold).detach().cpu().numpy().reshape(-1)))
        return label_pred

    @torch.no_grad()
    def soft_predict(self, xyz):
        self.model.eval()
        data_loader = DataLoader(xyz, batch_size=128, shuffle=False)
        label_pred = np.empty(0, dtype=float)
        for xyz_batch in data_loader:
            xyz_batch = xyz_batch.to(self.device)
            label_pred = np.concatenate((label_pred, self.model(xyz_batch).detach().cpu().numpy().reshape(-1)))
        return label_pred

    @torch.no_grad()
    def refine_sampling(self, threshold=0.5):
        self.model.eval()
        size_before = len(self.dataset)

        # next epoch will be with the refined dataset
        self.epochs_with_refine.append(self.total_epochs + 1)

        errored_xyz, _ = self.get_train_errors()

        def predictor(xyz):
            xyz = torch.from_numpy(xyz)
            xyz = xyz.to(self.device)
            return self.model(xyz) > threshold

        self.dataset.refine_cells(predictor)
        print(f'refine_sampling before={size_before}, after={len(self.dataset)}')

    @torch.no_grad()
    def get_train_errors(self, threshold=0.5):
        self.model.eval()
        errored_xyz = np.empty((0, 3), dtype=bool)
        errored_labels = np.empty((0, 1), dtype=bool)

        for xyz, label in self.data_loader:
            xyz, label = xyz.to(self.device), label.to(self.device)

            label_pred = self.model(xyz) > threshold
            errors = (label != label_pred).view(-1)

            errored_xyz = np.concatenate((errored_xyz, xyz[errors].detach().cpu().numpy()))
            errored_labels = np.concatenate((errored_labels, label[errors].detach().cpu().numpy()))

        return errored_xyz, errored_labels.reshape(-1)
