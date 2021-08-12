import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from Renderer import Renderer2
from Resterizer import rasterizer_factory


class RasterizedCslDataset(Dataset):
    def __init__(self, csl, sampling_resolution=(256, 256), margin=0.2, transform=None, target_transform=None):
        self.csl = csl
        samples = []
        for plane in csl.planes:
            if not plane.is_empty:  # todo add rasteresation to empty planes
                samples += rasterizer_factory(plane).get_rasterized(sampling_resolution, margin)

        self.labels_per_plane, self.xyz_per_plane = zip(*samples)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_per_plane) * len(self.labels_per_plane[0])

    def __getitem__(self, idx):
        i = idx % len(self.labels_per_plane)
        j = int(idx / len(self.labels_per_plane))

        xyz = self.xyz_per_plane[i][j]
        label = [1.0] if self.labels_per_plane[i][j] else [0.0]
        #label = [1] if xyz[0]+xyz[1]+xyz[2] > 0 else [-1]

        if self.transform:
            xyz = self.transform(xyz)
        if self.target_transform:
            label = self.target_transform(label)

        return xyz, label


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
    def __init__(self):
        self.save_path = "traind_model.pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.model = NaiveNetwork().to(self.device)
        self.model.double()

        print(self.model)

        self.loss_fn = None
        self.train_losses = []
        self.optimizer = None
        self.data_loader = None

        self.traning_ready = False
        self.total_epoches = 0

    def _train_epoch(self):
        assert self.traning_ready

        running_loss = 0.0
        size = len(self.data_loader.dataset)
        for batch, (xyz, label) in enumerate(self.data_loader):
            xyz, label = xyz.to(self.device), label.to(self.device)

            # Compute prediction error
            label_pred = self.model(xyz)
            #print(f"{label_pred.shape}, {label.shape}")
            loss = self.loss_fn(label_pred, label)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(xyz)
                print(f"\tloss: {loss:>7f}, running: {running_loss}  [{current:>5d}/{size:>5d}]")
        print(f"running loss for epoch: {running_loss}")
        self.train_losses.append(running_loss)

    def prepere_for_training(self, csl, sampling_resolution=(255, 255), margin=0.2, lr=1e-2):
        dataset = RasterizedCslDataset(csl, sampling_resolution=sampling_resolution, margin=margin,
                                       target_transform=torch.tensor, transform=torch.tensor)
        self.data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        self.model.init_weights()
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.traning_ready = True

        for xyz, label in self.data_loader:
            print("Shape of X [N, C, H, W]: ", xyz.shape)
            print("Shape of label: ", label.shape, label.dtype)
            break
        return self

    def train_network(self, epochs=30):
        self.model.train()
        for epoch in range(epochs):
            print(f"\n\nEpoch {self.total_epoches} [{epoch}/{epochs}]\n-------------------------------")
            self._train_epoch()
            self.total_epoches += 1
        print("Done!")
        return self

    def show_train_losses(self):
        plt.bar(range(len(self.train_losses)), self.train_losses)
        plt.show()
        return self

    def load_from_disk(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')) )
        self.model.eval()
        return self

    def save_to_disk(self):
        torch.save(self.model.state_dict(), self.save_path)
        return self

    @torch.no_grad()
    def predict(self, xyz):
        self.model.eval()
        xyz = torch.from_numpy(xyz).to(self.device)
        label_pred = self.model(xyz)
        return label_pred.detach().cpu().numpy().reshape(-1)