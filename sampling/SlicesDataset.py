import pickle

import torch
from Helpers import *
from torch.utils.data import Dataset

from sampling.Rasterizer import slices_rasterizer_factory


class SlicesDataset(Dataset):
    def __init__(self, csl, cells):
        self.csl = csl

        self.cells = cells

    @classmethod
    def from_cells(cls, csl, cells):
        return cls(csl, np.array(cells))

    @classmethod
    def from_csl(cls, csl, pool, hp, gen):
        cells = []
        for plane in csl.planes:
            # since we concat datasets - rester empty planes only once
            if not plane.is_empty or gen == 0:
                cells += slices_rasterizer_factory(plane, hp).get_rasterazation_cells(gen)

        cells = np.array(cells)

        # calculate density in pool
        # ret = pool.imap_unordered(lambda cell: cell.density, cells)
        return cls(csl, cells)

    def __len__(self):
        return self.cells.size

    def __getitem__(self, idx):
        cell = self.cells[idx]
        xyz = torch.tensor(cell.xyz)
        density = torch.tensor([cell.density])

        return xyz, density

    def pickle(self, file_name):
        pickle.dump(self.cells, open(file_name, 'wb'))

    def to_ply(self, file_name):
        header = f'ply\nformat ascii 1.0\nelement vertex {len(self.cells)}\n' \
                 f'property float x\nproperty float y\nproperty float z\n' \
                 f'property int generation\nproperty float quality\n' \
                 f'element face 0\nproperty list uchar int vertex_index\nend_header\n'

        with open(file_name, 'w') as f:
            f.write(header)
            for cell in self.cells:
                f.write('{:.10f} {:.10f} {:.10f} {} {:.10f}\n'.format(*cell.xyz, cell.generation, cell.density))