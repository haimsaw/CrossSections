from torch.utils.data import Dataset
import numpy as np

class RasterizedCslBoundaryDataset(Dataset):
    def __init__(self, csl, n_samples_per_edge, transform=None, target_transform=None):
        self.csl = csl

        # todo n_samples_per_edge should be per plane?

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    verts = plane.vertices[connected_component.vertices_indices]
                    verts1 = verts[1:] + verts[0:1]
                    normals = np.cross(verts1 - verts, plane.normal)

                    for p0, p1, normal in zip(verts, verts1, normals):
                        pass


        self.transform = transform
        self.target_transform = target_transform

    def len(self):
        pass

    def __getitem__(self, idx):
        pass
        # return xyz, normal

