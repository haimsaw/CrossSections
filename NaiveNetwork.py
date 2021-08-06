import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torch import nn


class RasterizedCslDataset(Dataset):
    def __init__(self, csl, sampling_resolution=(255, 255), margin=0.2):
        self.csl = csl
        samples = [plane.rasterizer.get_rasterized(sampling_resolution, margin) for plane in self.csl.planes
                                 if len(plane.vertices) > 0]  # todo add rasteresation to empty planes

        self.labels_per_plane, self.xyz_per_plane = zip(*samples)

    def __len__(self):
        return len(self.labels_per_plane) * len(self.labels_per_plane[0])

    def __getitem__(self, idx):

        i = idx % len(self.labels_per_plane)
        j = int(idx / len(self.labels_per_plane))

        return self.xyz_per_plane[i][j], int(self.labels_per_plane[i][j])


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


def train(dataloader, model, loss_fn, optimizer, device):
    running_loss = 0.0
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        y_pred = model(x)
        loss = loss_fn(y_pred, y.view(-1, 1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch * len(x)
        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return running_loss


def run_naive_network(csl, sampling_resolution=(255, 255), margin=0.2, epoches=30):
    dataset = RasterizedCslDataset(csl, sampling_resolution=sampling_resolution, margin=margin)
    dataloader = DataLoader(dataset, batch_size=128)
    for X1, y in dataloader:
        print("Shape of X [N, C, H, W]: ", X1.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = NaiveNetwork().to(device)
    model.double()
    print(model)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    losses = []

    for epoch in range(epoches):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        losses.append(train(dataloader, model, loss_fn, optimizer, device))
    torch.save(model.state_dict(), "traind_model.pt")
    plt.plot(losses)
    plt.show()
    print("Done!")
    return model


def get_saved_model():
    model = NaiveNetwork()
    model.load_state_dict(torch.load("traind_model.pt"))
    model.eval()
    return model

