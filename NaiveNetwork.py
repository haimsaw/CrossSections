import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torch import nn


class RasterizedCslDataset(Dataset):
    def __init__(self, csl, sampling_resolution=(255, 255), margin=0.2, transform=None, target_transform=None):
        self.csl = csl
        samples = [plane.rasterizer.get_rasterized(sampling_resolution, margin) for plane in self.csl.planes
                   if len(plane.vertices) > 0]  # todo add rasteresation to empty planes

        self.labels_per_plane, self.xyz_per_plane = zip(*samples)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_per_plane) * len(self.labels_per_plane[0])

    def __getitem__(self, idx):
        i = idx % len(self.labels_per_plane)
        j = int(idx / len(self.labels_per_plane))

        xyz = self.xyz_per_plane[i][j]
        label = [int(self.labels_per_plane[i][j])]

        if self.transform:
            xyz = self.transform(xyz)
        if self.target_transform:
            label = self.target_transform(label)

        return xyz, label


class NaiveNetwork(nn.Module):
    def __init__(self):
        super(NaiveNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear_relu(x)
        return x


class NetworkManager:
    def __init__(self):
        self.save_path = "traind_model.pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.model = NaiveNetwork().to(self.device)
        self.model.double()
        print(self.model)

        # loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.L1Loss()

    def _train_epoch(self, data_loader, optimizer):
        running_loss = 0.0
        size = len(data_loader.dataset)
        for batch, (xyz, label) in enumerate(data_loader):
            xyz, label = xyz.to(self.device), label.to(self.device)

            # Compute prediction error
            label_pred = self.model(xyz)
            loss = self.loss_fn(label_pred, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch * len(xyz)
            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(xyz)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return running_loss

    def train_network(self, csl, sampling_resolution=(255, 255), margin=0.2, epochs=30):
        dataset = RasterizedCslDataset(csl, sampling_resolution=sampling_resolution, margin=margin,
                                       target_transform=torch.tensor, transform=torch.tensor)
        data_loader = DataLoader(dataset, batch_size=128)

        for xyz, label in data_loader:
            print("Shape of X [N, C, H, W]: ", xyz.shape)
            print("Shape of label: ", label.shape, label.dtype)
            break

        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        losses = []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            losses.append(self._train_epoch(data_loader, optimizer))
        torch.save(self.model.state_dict(), self.save_path)
        plt.plot(losses)
        plt.show()
        print("Done!")
        return self

    def load_from_disk(self):
        self.model.load_state_dict(torch.load("traind_model.pt"))
        self.model.eval()
        return self

    def predict(self, xyz):
        xyz = torch.from_numpy(xyz).to(self.device)
        lable_pred = self.model(xyz)
