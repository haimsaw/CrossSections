import torch
from torch.utils.data import Dataset
import numpy as np

from Helpers import get_xyzs_in_octant


class ContourDataset(Dataset):
    def __init__(self, csl, n_samples_per_edge):
        self.csl = csl

        self.xyzs = np.empty(shape=(0, 3))
        self.normals = np.empty(shape=(0, 3))
        self.tangents = np.empty(shape=(0, 3))
        self.slice_normals = np.empty(shape=(0, 3))

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    verts = plane.vertices[connected_component.vertices_indices]
                    verts1 = np.concatenate((verts[1:], verts[0:1]))

                    # no need to invert order for holes since they ordered cw
                    tangents = verts1 - verts
                    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

                    normals = np.cross(verts1 - verts, plane.normal)
                    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

                    for p0, p1, normal, edge in zip(verts, verts1, normals, tangents):

                        # todo n_samples_per_edge should be per plane?
                        self.xyzs = np.concatenate((self.xyzs, np.linspace(p0, p1, n_samples_per_edge, endpoint=False)))
                        self.normals = np.concatenate((self.normals, np.repeat([normal], n_samples_per_edge, axis=0)))
                        self.tangents = np.concatenate((self.tangents, np.repeat([edge], n_samples_per_edge, axis=0)))
                        self.slice_normals = np.concatenate((self.slice_normals, np.repeat([plane.normal], n_samples_per_edge, axis=0)))

    def __len__(self):
        return len(self.xyzs)

    def __getitem__(self, idx):
        xyz = torch.tensor(self.xyzs[idx])
        normal = torch.tensor(self.normals[idx])
        tangent = torch.tensor(self.tangents[idx])
        slice_normal = torch.tensor(self.slice_normals[idx])

        return xyz, normal, tangent, slice_normal


class ContourDatasetFake(Dataset):
    def __init__(self, csl, n_samples_per_edge, transform=None, target_transform=None, edge_transform=None):
        self.radius = 0.4

        self.xyzs = np.random.randn(1000, 3)
        self.xyzs = (self.xyzs / np.linalg.norm(self.xyzs, axis=0)) * self.radius

        self.transform = transform
        self.normal_transform = target_transform
        self.edge_transform = edge_transform

    def __len__(self):
        return len(self.xyzs)

    def __getitem__(self, idx):
        xyz = self.xyzs[idx]
        normal = xyz / np.linalg.norm(xyz)

        tangent = np.array([0, 1, -normal[1]/normal[2]])
        tangent /= np.linalg.norm(tangent)

        if self.transform:
            xyz = self.transform(xyz)
        if self.normal_transform:
            normal = self.normal_transform(normal)
        if self.edge_transform:
            tangent = self.edge_transform(tangent)

        return xyz, normal, tangent