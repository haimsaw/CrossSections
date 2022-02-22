import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Modules import *
from Helpers import *
from abc import ABCMeta, abstractmethod


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
        self.hp = None

        self.optimizer = None
        self.lr_scheduler = None
        self.scheduler_step = 0

        self.slices_data_loader = None
        self.contour_data_loader = None

        self.total_epochs = 0
        self.train_losses = []

        self.is_training_ready = False

    def _get_losses(self, slices_xyzs, slices_density, contour_xyzs, normals_on_contour, tangents_on_contour):
        slices_xyzs.requires_grad_(True)
        contour_xyzs.requires_grad_(True)

        slices_density_pred = self.module(slices_xyzs)
        contour_labels_pred = self.module(contour_xyzs)

        # todo haim grad_outputs
        d_xyzs_d_density = torch.autograd.grad(slices_density_pred.sum(), [slices_xyzs], create_graph=True)[0]
        contour_xyzs_grad = torch.autograd.grad(contour_labels_pred.sum(), [contour_xyzs], create_graph=True)[0]

        losses = {}

        # density - zero inside one outside
        # bce_loss has a sigmoid layer build in
        if self.hp.density_lambda > 0:
            losses['density'] = self.bce_loss(slices_density_pred, slices_density) * self.hp.density_lambda

        # grad(f(x)) = 1 everywhere (eikonal)
        if self.hp.eikonal_lambda > 0:
            losses['eikonal'] = (d_xyzs_d_density.norm(dim=-1) - 1).abs().mean() * self.hp.eikonal_lambda

        # f(x) = 0 on contour
        if self.hp.contour_val_lambda > 0:
            losses['contour_val'] = contour_labels_pred.abs().mean() * self.hp.contour_val_lambda

        # grad(f(x))*normal = 1 on contour
        if self.hp.contour_normal_lambda > 0:
            losses['contour_normal'] = ((contour_xyzs_grad * normals_on_contour).sum(dim=-1) - 1).abs().mean() * self.hp.contour_normal_lambda

        # grad(f(x)) * contour_tangent = 0 on contour
        if self.hp.contour_tangent_lambda > 0:
            losses['contour_tangent'] = ((contour_xyzs_grad * tangents_on_contour).sum(dim=-1)).abs().mean() * self.hp.contour_tangent_lambda

        return losses

    def _train_epoch(self, epoch):
        assert self.is_training_ready
        if self.verbose:
            print(f"\n\nEpoch {self.total_epochs} [{epoch}]\n-------------------------------")

        running_loss = 0.0
        running_losses = {}
        size = len(self.slices_data_loader.dataset)

        for batch, ((slices_xyzs, slices_density), (contour_xyzs, contour_normals, contour_tangents)) in enumerate(zip(self.slices_data_loader, self.contour_data_loader)):
            slices_xyzs, slices_density = slices_xyzs.to(self.device), slices_density.to(self.device)
            contour_xyzs, contour_normals, contour_tangents = contour_xyzs.to(self.device), contour_normals.to(self.device), contour_tangents.to(self.device)

            # _get_loss should call self.optimizer.zero_grad() at the end
            losses = self._get_losses(slices_xyzs, slices_density, contour_xyzs, contour_normals, contour_tangents)
            running_losses = {k: losses.get(k, torch.tensor([0])).item() + running_losses.get(k, 0) for k in set(losses)}

            self.optimizer.zero_grad()
            total_loss = sum(losses.values())
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item() * len(slices_xyzs)
            if self.verbose and batch % 1000 == 0:
                bce_loss, current = total_loss.item(), batch * len(slices_xyzs)
                print(f"\tloss: {total_loss:>7f}, running: {running_loss}  [{current:>5d}/{size:>5d}]")

        if epoch > 0 and epoch % self.scheduler_step == 0:
            self.lr_scheduler.step()

        total_loss = running_loss / size
        print(f'epoch={epoch} losses={running_losses}')

        if self.verbose:
            print(f"\tloss for epoch: {total_loss}")
        self.train_losses.append(total_loss)

    def prepare_for_training(self, slices_dataset, slices_sampler, contour_dataset, contour_sampler, hp):
        self.slices_data_loader = DataLoader(slices_dataset, batch_size=128, sampler=slices_sampler)
        self.contour_data_loader = DataLoader(contour_dataset, batch_size=128, sampler=contour_sampler)

        self.module.init_weights()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.hp = hp

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler_step = hp.scheduler_step

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

        for xyzs, label in self.slices_data_loader:
            xyzs, label = xyzs.to(self.device), label.to(self.device)

            # xyzs, label_pred = self.hard_predict(xyzs, threshold)
            label_pred = self.module(xyzs) > threshold  # todo use self.hard_predict (not returning a tensor)

            errors = (label != label_pred).view(-1)

            errored_xyzs = np.concatenate((errored_xyzs, xyzs[errors].detach().cpu().numpy()))
            errored_labels = np.concatenate((errored_labels, label[errors].detach().cpu().numpy()))

        return errored_xyzs, errored_labels.reshape(-1)
