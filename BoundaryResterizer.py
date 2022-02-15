from torch.utils.data import Dataset
import numpy as np


class RasterizedCslBoundaryDataset(Dataset):
    def __init__(self, csl, n_samples_per_edge, transform=None, target_transform=None):
        self.csl = csl

        self.xyzs = np.empty(shape=(0, 3))
        self.normals = np.empty(shape=(0, 3))

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    verts = plane.vertices[connected_component.vertices_indices]
                    verts1 = np.concatenate((verts[1:], verts[0:1]))

                    # no need to invert order for holes since they ordered cw
                    normals = np.cross(verts1 - verts, plane.normal)

                    for p0, p1, normal in zip(verts, verts1, normals):
                        normal /= np.linalg.norm(normal)

                        # todo n_samples_per_edge should be per plane?
                        self.xyzs = np.concatenate((self.xyzs, np.linspace(p0, p1, n_samples_per_edge, endpoint=False)))
                        self.normals = np.concatenate((self.normals, np.repeat([normal], n_samples_per_edge, axis=0)))

        self.transform = transform
        self.normal_transform = target_transform

    def len(self):
        return len(self.xyzs)

    def __getitem__(self, idx):
        xyz = self.xyzs[idx]
        normal = self.normals[idx]

        if self.transform:
            xyz = self.transform(xyz)
        if self.normal_transform:
            normal = self.normal_transform(normal)

        return xyz, normal
