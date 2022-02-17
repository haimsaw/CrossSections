from torch.utils.data import Dataset
import numpy as np


class BoundaryDataset(Dataset):
    def __init__(self, csl, n_samples_per_edge, transform=None, target_transform=None, edge_transform=None):
        self.csl = csl

        self.transform = transform
        self.normal_transform = target_transform
        self.edge_transform = edge_transform

        self.xyzs = np.empty(shape=(0, 3))
        self.normals = np.empty(shape=(0, 3))
        self.edges = np.empty(shape=(0, 3))

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    verts = plane.vertices[connected_component.vertices_indices]
                    verts1 = np.concatenate((verts[1:], verts[0:1]))

                    # no need to invert order for holes since they ordered cw
                    edges = verts1 - verts
                    edges = edges / np.linalg.norm(edges, axis=1)[:, np.newaxis]

                    normals = np.cross(verts1 - verts, plane.normal)
                    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

                    for p0, p1, normal, edge in zip(verts, verts1, normals, edges):

                        # todo n_samples_per_edge should be per plane?
                        self.xyzs = np.concatenate((self.xyzs, np.linspace(p0, p1, n_samples_per_edge, endpoint=False)))
                        self.normals = np.concatenate((self.normals, np.repeat([normal], n_samples_per_edge, axis=0)))
                        self.edges = np.concatenate((self.edges, np.repeat([edge], n_samples_per_edge, axis=0)))

    def __len__(self):
        return len(self.xyzs)

    def __getitem__(self, idx):
        xyz = self.xyzs[idx]
        normal = self.normals[idx]
        edge = self.edges[idx]

        if self.transform:
            xyz = self.transform(xyz)
        if self.normal_transform:
            normal = self.normal_transform(normal)
        if self.edge_transform:
            edge = self.edge_transform(edge)

        return xyz, normal, edge