from Helpers import *
import numpy as np
import matplotlib.pyplot as plt

from SlicesDataset import slices_rasterizer_factory, INSIDE_LABEL
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.animation as animation
import polyscope as ps

# region polyscope

# endregion


def render_mid_res(csl, trainer, samplig_res_3d):
    verts = np.empty((0, 3))
    edges = np.empty((0, 2))

    for plane in csl.planes:

        plane_vert_start = len(verts)
        verts = np.concatenate((verts, plane.vertices))

        for cc in plane.connected_components:
            e1 = cc.vertices_indices + plane_vert_start
            e2 = np.concatenate((cc.vertices_indices[1:], cc.vertices_indices[0:1])) + plane_vert_start

            edges = np.concatenate((edges, np.stack((e1, e2)).T))

    xyzs = get_xyzs_in_octant(None, samplig_res_3d)
    labels = trainer.hard_predict(xyzs)

    ps.init()
    # ps.set_ground_plane_height_factor(-0.25)
    # ps.set_transparency_mode('pretty')
    # ps.look_at((0., 0., -2.5), (0., 0., 0.))

    ps.set_ground_plane_mode("none")
    model = ps.register_point_cloud("model", xyzs[labels], material="candy", transparency=0.4)
    scene = ps.register_curve_network(f"scene", verts, edges, material="candy")

    scene.set_radius(scene.get_radius()/1.5)
    model.set_radius(model.get_radius() * 1)

    ps.set_screenshot_extension(".png")
    ps.screenshot()

    ps.show()


class RendererPoly:

    @staticmethod
    def init():
        ps.init()
        # ps.set_ground_plane_height_factor(-0.25)
        # ps.set_transparency_mode('pretty')
        ps.set_ground_plane_mode("none")

    @staticmethod
    def add_scene(csl):
        verts = np.empty((0, 3))
        edges = np.empty((0, 2))

        for plane in csl.planes:

            plane_vert_start = len(verts)
            verts = np.concatenate((verts, plane.vertices))

            for cc in plane.connected_components:
                e1 = cc.vertices_indices + plane_vert_start
                e2 = np.concatenate((cc.vertices_indices[1:], cc.vertices_indices[0:1])) + plane_vert_start

                edges = np.concatenate((edges, np.stack((e1, e2)).T))
        ps_net = ps.register_curve_network(f"scene", verts, edges)

    @staticmethod
    def add_model_hard_prediction(network_manager, sampling_resolution_3d):
        xyzs = get_xyzs_in_octant(None, sampling_resolution_3d)
        labels = network_manager.hard_predict(xyzs)

        ps_cloud = ps.register_point_cloud("model", xyzs[labels], material="candy", transparency=0.5)

    @staticmethod
    def add_mesh(vertices, faces):
        ps.register_surface_mesh("my mesh", vertices, faces)

    @staticmethod
    def show():
        ps.show()

# region 3d


class Renderer3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 15))
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.description = []

    def add_scene(self, csl):
        self.description.append('scene')
        # colors = [[random(), random(), random()] for _ in range(csl.n_labels + 1)]

        for plane in csl.planes:
            if not plane.is_empty:
                for connected_component in plane.connected_components:
                    vertices = plane.vertices[connected_component.vertices_indices]
                    vertices[-1] = vertices[0]
                    alpha = 1 if connected_component.is_hole else 0.5
                    # ax.plot_trisurf(*vertices.T, color='green', alpha=alpha)
                    # todo haim facecolors?
                    self.ax.plot(*vertices.T, color='green')
                    # ax.plot_surface(*vertices.T, color='green')

    def add_dataset(self, dataset):
        self.description.append('dataset')

        xyzs = np.array([xyz.detach().numpy() for xyz, label in dataset if label == INSIDE_LABEL])
        self.ax.scatter(*xyzs.T, color="blue")

    def add_rasterized_scene(self, csl, sampling_resolution_2d, sampling_margin, show_empty_planes=True, show_outside_shape=False, alpha=0.1):
        self.description.append('rasterized_scene')

        for plane in csl.planes:
            cells = slices_rasterizer_factory(plane).get_rasterazation_cells(sampling_resolution_2d, sampling_margin)
            mask = np.array([cell.density <= 0.5 for cell in cells])
            xyzs = np.array([cell.xyz for cell in cells])

            if not plane.is_empty:
                self.ax.scatter(*xyzs[mask].T, color="blue", alpha=alpha)
                if show_outside_shape:
                    self.ax.scatter(*xyzs[np.logical_not(mask)].T, color="gray", alpha=alpha)
            elif show_empty_planes:
                self.ax.scatter(*xyzs.T, color="purple", alpha=alpha/2)

    def add_model_hard_prediction(self, network_manager, sampling_resolution_3d, alpha=0.05, octant=None):
        self.description.append('hard_prediction')
        xyzs = get_xyzs_in_octant(octant, sampling_resolution_3d)
        labels = network_manager.hard_predict(xyzs)

        self.ax.scatter(*xyzs[labels].T, alpha=alpha, color='blue')

    def add_mesh(self, my_mesh, alpha=0.2):
        self.description.append('data')
        collection = Poly3DCollection(my_mesh.vectors, alpha=alpha)
        collection.set_edgecolor('b')
        self.ax.add_collection3d(collection)

    def add_model_errors(self, network_manager):
        self.description.append('model_errors')
        errored_xyz, errored_labels = network_manager.get_train_errors()
        self.ax.scatter(*errored_xyz[errored_labels == 1].T, color="purple")
        self.ax.scatter(*errored_xyz[errored_labels == 0].T, color="red")

    def add_model_grads(self, network_manager, xyzs, alpha=1, length=0.1, neg=False):
        self.description.append('grads')

        grads = network_manager.grad_wrt_input(xyzs)
        if neg:
            grads = -1 * grads

        self.ax.quiver(*xyzs.T, *grads.T, color='black', alpha=alpha, length=length, normalize=True)

    '''
    def add_contour_normals(self, csl, n_samples_per_edge=1, alpha=0.5, length=0.1):
        dataset = ContourDataset(csl, n_samples_per_edge)
        xyzs, normals, _ = zip(*list(dataset))
        self.ax.quiver(*np.array(xyzs).T, *np.array(normals).T, color='black', alpha=alpha, length=length, normalize=True)

    def add_contour_tangents(self, csl, n_samples_per_edge=1, alpha=0.5, length=0.1):
        dataset = ContourDataset(csl, n_samples_per_edge)
        xyzs, _, tangents = zip(*list(dataset))
        self.ax.quiver(*np.array(xyzs).T, *np.array(tangents).T, color='black', alpha=alpha, length=length, normalize=True)
    '''

    def show(self):
        plt.show()

    def save_animation(self, save_path, level, elevs=(-30,)):

        for elev in elevs:
            def rotate(angle):
                self.ax.view_init(elev=elev, azim=angle)

            name = save_path + '_'.join(self.description) + f'_l{level}' + f'_elev{elev}' + '.gif'
            rot_animation = animation.FuncAnimation(self.fig, rotate, frames=range(0, 360, 5), interval=150)
            rot_animation.save(name, writer='imagemagick')
            # rot_animation.save(name,dpi=80, writer=animation.ImageMagickWriter)

            #rot_animation.save(name, writer='pillow')
        self.ax.view_init()

# endregion


# region 2d

class Renderer2D:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 15))
        self.ax = plt.axes()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        self.description = []

    def draw_rasterized_plane(self, plane, resolution=(256, 256), margin=0.2):
        cells = slices_rasterizer_factory(plane).get_rasterazation_cells(resolution, margin)
        is_inside = np.array([cell.density == INSIDE_LABEL for cell in cells])
        xyz = np.array([cell.pixel_center for cell in cells])
        self.ax.scatter(*xyz[is_inside].T, color='red')
        self.ax.scatter(*xyz[np.logical_not(is_inside)].T, color='blue')
        self.description.append("draw_rasterized_plane")

    def draw_plane_verts(self, plane):
        verts, _ = plane.pca_projection
        for component in plane.connected_components:
            cc_verts = verts[component.vertices_indices]
            self.ax.scatter(*cc_verts.T, color='orange' if component.is_hole else 'black')
            for i, vert in enumerate(cc_verts):
                self.ax.annotate(str(i), vert)

        self.ax.scatter([0], [0], color='red')
        self.description.append("draw_plane_verts")

    def draw_plane(self, plane):
        if not plane.is_empty:
            verts, _ = plane.pca_projection
            for component in plane.connected_components:
                self.ax.plot(*verts[component.vertices_indices].T, color='orange' if component.is_hole else 'black')
            self.description.append("draw_plane")

    def heatmap(self, sampling_resolution_2d, network_manager, around_ax, dist, add_grad):
        assert around_ax in (0, 1, 2)
        self.description.append(f"heatmap around_ax_{around_ax}_at{str(dist)}_{'grad' if add_grad else ''}")

        extent = -1, 1, -1, 1
        sampling_resolution_3d = np.insert(sampling_resolution_2d, around_ax, 1)
        oct = np.array([[1.0]*3, [-1.0]*3])
        oct[:, around_ax] = dist

        xyzs = get_xyzs_in_octant(oct, sampling_resolution_3d)
        labels = network_manager.soft_predict(xyzs)

        pos = self.ax.imshow(labels.reshape(sampling_resolution_2d).T, origin='lower',
                             cmap='plasma', extent=extent, interpolation='bilinear')

        self.fig.colorbar(pos)

        if add_grad:
            grads_2d = np.delete(network_manager.grad_wrt_input(xyzs), around_ax, axis=1)
            xys = np.delete(xyzs, around_ax, axis=1)
            self.ax.quiver(*xys.T, *grads_2d.T, color='black', alpha=1.0)

    def save(self, save_path):
        title = '_'.join(self.description)
        name = save_path + title + '.svg'
        self.fig.suptitle(title)
        plt.savefig(name)

    def show(self):
        title = '_'.join(self.description)
        self.fig.suptitle(title, fontsize=16)
        plt.show()

    def clear(self):
        self.fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

    '''
    def save_animation(self, save_path, level, elevs=(-30,)):

        for elev in elevs:
            def rotate(angle):
                self.ax.view_init(elev=elev, azim=angle)

            name = save_path + '_'.join(self.description) + f'_l{level}' + f'_elev{elev}' + '.gif'
            rot_animation = animation.FuncAnimation(self.fig, rotate, frames=range(0, 360, 5), interval=150)
            rot_animation.save(name, writer='imagemagick')
            # rot_animation.save(name,dpi=80, writer=animation.ImageMagickWriter)

            #rot_animation.save(name, writer='pillow')
        self.ax.view_init()
    '''

# endregion
