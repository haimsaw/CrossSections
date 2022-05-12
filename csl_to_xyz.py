import numpy as np
from matplotlib import pyplot as plt

from hp import get_csl, HP

def csl_to_xyz():
    save_path = "artifacts/xyz/"
    hp = HP()
    n_points_per_edge = 5

    csl = get_csl(hp.bounding_planes_margin, save_path)
    pts = []
    for plane in csl.planes():
        for cc in plane.connected_components:
            verts = plane.vertices[cc.vertices_indices]
            verts2 = np.concatenate((verts[1:], verts[0:1]))
            for e1, e2  in zip(verts, verts2):
                pts += [e1*d + e2(1-d) for d in np.linspace(0, 1, n_points_per_edge)]

    pts = np.array(pts)
    ax = plt.axes()
    ax.scatter(*pts.T, color='red')
    plt.show()

if __name__ == "__main__":
    csl_to_xyz()