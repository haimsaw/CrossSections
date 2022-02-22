from datetime import datetime
import json

from CSL import *
from ContourRasterizer import ContourDatasetFake
from DomainResterizer import DomainDataset, DomainDatasetFake
from Renderer import *
from Mesher import *
from OctnetTree import *


def get_csl(bounding_planes_margin):
    csl = CSL("csl-files/ParallelEight.csl")
    # csl = CSL("csl-files/ParallelEightMore.csl")
    # csl = CSL("csl-files/SideBishop.csl")
    # csl = CSL("csl-files/Heart-25-even-better.csl")
    # csl = CSL("csl-files/Armadillo-23-better.csl")
    # csl = CSL("csl-files/Horsers.csl")
    # csl = CSL("csl-files/rocker-arm.csl")
    # csl = CSL("csl-files/Abdomen.csl")
    # csl = CSL("csl-files/Vetebrae.csl")
    # csl = CSL("csl-files/Skull-20.csl")
    # csl = CSL("csl-files/Brain.csl")

    csl.adjust_csl(bounding_planes_margin=bounding_planes_margin)
    return csl


class HP:
    def __init__(self):
        # sampling
        self.bounding_planes_margin = 0.05
        self.sampling_margin = 0.05  # same as bounding_planes_margin
        self.oct_overlap_margin = 0.25

        # resolutions
        self.root_sampling_resolution_2d = (32, 32)
        self.sampling_resolution_3d = (64, 64, 64)
        self.contour_sampling_resolution = 5

        # architecture
        self.num_embedding_freqs = 4
        self.hidden_layers = [64, 64, 64, 64, 64]
        self.is_siren = False

        # loss
        self.weight_decay = 1e-3  # l2 regularization

        self.density_lambda = 1

        self.eikonal_lambda = 0
        self.contour_val_lambda = 0
        self.contour_normal_lambda = 0
        self.contour_tangent_lambda = 0

        # training
        self.epochs = 10
        self.scheduler_step = 5
        self.lr = 1e-2

        # inference
        self.sigmoid_on_inference = False

        self.now = str(datetime.now())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def main():
    hp = HP()

    csl = get_csl(hp.bounding_planes_margin)
    
    '''
    renderer = Renderer3D()
    renderer.add_scene(csl)
    renderer.add_contour_tangents(csl)
    # renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    # renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    # renderer.add_rasterized_scene(csl, hp.root_sampling_resolution_2d, hp.sampling_margin, show_empty_planes=True, show_outside_shape=True)
    renderer.show()
    # '''

    print(f'loss: density={hp.density_lambda}, eikonal={hp.eikonal_lambda}, contour_val={hp.contour_val_lambda}, contour_normal={hp.contour_normal_lambda}, contour_tangent={hp.contour_tangent_lambda}')

    tree = OctnetTree(csl, hp.oct_overlap_margin, hp.hidden_layers, get_embedder(hp.num_embedding_freqs), hp.is_siren)

    # d2_res = [i * (2 ** (tree.depth + 1)) for i in hp.root_sampling_resolution_2d]
    domain_dataset = DomainDataset(csl, calc_density=hp.density_lambda>0, sampling_resolution=hp.root_sampling_resolution_2d, sampling_margin=hp.sampling_margin,
                                   target_transform=torch.tensor, transform=torch.tensor)
    contour_dataset = ContourDataset(csl, round(len(domain_dataset) / len(csl)),  # todo haim
                                      target_transform=torch.tensor, transform=torch.tensor, edge_transform=torch.tensor)

    # level 0:
    tree.prepare_for_training(domain_dataset, contour_dataset, hp)
    tree.train_network(epochs=hp.epochs)

    # tree.show_train_losses()

    try:
        mesh_mc = marching_cubes(tree, hp.sampling_resolution_3d, hp.sigmoid_on_inference)
        mesh_mc.save('output_mc.stl')

        renderer = Renderer3D()
        renderer.add_scene(csl)
        renderer.add_mesh(mesh_mc)
        renderer.show()
    except ValueError as e:
        print(e)
    finally:
        for dim in (0, 2):
            for dist in np.linspace(-1, 1, 3):
                renderer = Renderer2D()
                renderer.heatmap([100] * 2, tree, dim, dist, True, hp.sigmoid_on_inference)
                renderer.save('')
                renderer.clear()

    return

    #mesh_dc = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=True, use_sigmoid=hp.sig_on_inference)
    #mesh_dc.save('output_dc_grad.obj')

    #mesh_dc = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=False, use_sigmoid=hp.sig_on_inference)
    #mesh_dc.save('output_dc_no_grad.obj')

    # verts = mesh_mc.vectors.reshape(-1, 3).astype(np.double)
    # verts = 2 * mesh_dc.verts/hp.sampling_resolution_3d - 1

    verts = np.array(csl.all_vertices)
    print(verts.shape)
    verts = verts[np.random.choice(verts.shape[0], 200, replace=False)]

    renderer.add_domain_grads(tree, verts, alpha=1, length=0.15, neg=True)
    renderer.show()

    # level 1
    tree.prepare_for_training(domain_dataset, contour_dataset, hp.lr, hp.scheduler_step, hp.weight_decay)
    tree.train_network(epochs=hp.epochs)

    # level 2
    tree.prepare_for_training(domain_dataset, contour_dataset, hp.lr, hp.scheduler_step, hp.weight_decay)
    tree.train_network(epochs=hp.epochs)

    # mesh = marching_cubes(network_manager_root, hp.sampling_resolution_3d)
    # renderer = Renderer3D()
    # renderer.add_mesh(mesh)
    # renderer.add_scene(csl)
    # renderer.add_model_errors(network_manager_root)
    # renderer.show()


def draw_blending_errors(tree, xyzs, title):
    labels = tree.soft_predict(xyzs)
    print(f'max={max(labels)}, min={min(labels)}, n={len(labels)} depth={tree.depth}')

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_title(title)

    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.scatter(*xyzs[(labels != 1) & (labels != 0)].T, c=labels[(labels != 1) & (labels != 0)], alpha=0.2)
    ax.scatter(*xyzs.T, c=labels, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    main()

'''
todo
check if tree is helping or its just capacity 
optimization - id dansity lambda ==0 -> no need to calc dansity

read: https://lioryariv.github.io/volsdf/
        https://lioryariv.github.io/idr/
        https://arxiv.org/abs/2202.01999
        poassion reconstruction
        nural dc

loss: add in boundary grad*tangent = 0  or grad * normal = 0
          in boundary eikonal 
          grad =0 away from boundary 

serialize a tree (in case collab crashes)
increase sampling (in prev work he used 2d= 216, 3d=300)

Sheared weights / find a way to use symmetries 
reduce capacity of lower levels (make #params in each level equal)

smooth loss - sink horn 

Use sinusoidal activations (SIREN/SAPE)
Scale & translate each octant to fit [-1,1]^3

Use loss from the upper level to determine depth \ #epochs

nerfs literature review  
'''
