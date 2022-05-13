import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SlicesDataset import INSIDE_LABEL, OUTSIDE_LABEL

from Modules import *
from Helpers import *
from abc import ABCMeta, abstractmethod


class INetManager:

    def __init__(self, csl, verbose=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'device={self.device}')

        self.csl = csl
        self.verbose = verbose

    @abstractmethod
    def soft_predict(self, xyzs): raise NotImplementedError

    @torch.no_grad()
    def hard_predict(self, xyzs, threshold=0):
        # todo self.module.eval()
        soft_labels = self.soft_predict(xyzs)
        return soft_labels < threshold

