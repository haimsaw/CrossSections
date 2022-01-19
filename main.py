from datetime import datetime
import json
import os
import mcdc.utils_3d as utils_3d
import numpy as np

from CSL import *
from Renderer import *
from NetManager import *
from Mesher import *
from Helpers import *
from Modules import *
from stl import mesh as mesh2
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


def main():
    hp = {
        # sampling
        'bounding_planes_margin': 0.05,
        'sampling_margin': 0.05,  # same as bounding_planes_margin
        'oct_overlap_margin': 0.25,

        # resolutions
        'root_sampling_resolution_2d':  (32, 32),
        'sampling_resolution_3d': (64, 64, 64),

        # architecture
        'num_embedding_freqs': 4,
        'hidden_layers': [64, 64, 64, 64, 64],
        'is_siren': False,

        # loss
        'eikonal_lambda': 1e-3,

        # training
        'epochs': 5,
        'scheduler_step': 5,
        'lr': 1e-2,
        'weight_decay': 1e-3,  # l2 regularization

        # inference
        'sig_on_inference': False,  # True

        'now': str(datetime.now()),
    }

    csl = get_csl(hp['bounding_planes_margin'])

    '''
    renderer = Renderer3D()
    renderer.add_scene(csl)
    # renderer.add_mesh(mesh2.Mesh.from_file('G:\\My Drive\\DeepSlice\\examples 2021.11.24\\wavelets\\Abdomen\\mesh-wavelets_1_level.stl'), alpha=0.05)
    #renderer.add_mesh(mesh2.Mesh.from_file('C:\\Users\\hasawday\\Downloads\\mesh-wavelets_1_level (9).stl'), alpha=0.05)

    # renderer.add_rasterized_scene(csl, hp['root_sampling_resolution_2d'], hp['sampling_margin'], show_empty_planes=True, show_outside_shape=True)
    renderer.show()
    '''

    tree = OctnetTree(csl, hp['oct_overlap_margin'], hp['hidden_layers'], get_embedder(hp['num_embedding_freqs']), hp['is_siren'])

    # d2_res = [i * (2 ** (tree.depth + 1)) for i in hp['root_sampling_resolution_2d']]
    dataset = RasterizedCslDataset(csl, sampling_resolution=hp['root_sampling_resolution_2d'], sampling_margin=hp['sampling_margin'],
                                   target_transform=torch.tensor, transform=torch.tensor)

    # level 0:
    tree.prepare_for_training(dataset, hp['lr'], hp['scheduler_step'], hp['weight_decay'], hp['eikonal_lambda'])
    tree.train_network(epochs=hp['epochs'])

    mesh_dc = dual_contouring(tree, hp['sampling_resolution_3d'], use_grads=True, use_sigmoid=hp['sig_on_inference'])
    mesh_dc.save('output_dc_grad.obj')

    mesh_dc = dual_contouring(tree, hp['sampling_resolution_3d'], use_grads=False, use_sigmoid=hp['sig_on_inference'])
    mesh_dc.save('output_dc_no_grad.obj')

    #mesh_mc = marching_cubes(tree, hp['sampling_resolution_3d'])
    #mesh_mc.save('output_mc.obj')

    for dist in np.linspace(-1, 1, 5):

        renderer = Renderer2D()
        renderer.heatmap([100]*2, tree, 2, dist, True, hp['sig_on_inference'])
        renderer.save('')

    return

    renderer = Renderer3D()
    renderer.add_scene(csl)
    #renderer.add_mesh(mesh_mc)

    # verts = mesh_mc.vectors.reshape(-1, 3).astype(np.double)
    #verts = 2 * mesh_dc.verts/hp['sampling_resolution_3d'] - 1

    verts = np.array(csl.all_vertices)
    print(verts.shape)
    verts = verts[np.random.choice(verts.shape[0], 200, replace=False)]

    renderer.add_grads(tree, verts, alpha=1, length=0.15, neg=True)
    renderer.show()

    # level 1
    tree.prepare_for_training(dataset, hp['lr'], hp['scheduler_step'], hp['weight_decay'], hp['eikonal_lambda'])
    tree.train_network(epochs=hp['epochs'])



    # level 2
    tree.prepare_for_training(dataset, hp['lr'], hp['scheduler_step'], hp['weight_decay'], hp['eikonal_lambda'])
    tree.train_network(epochs=hp['epochs'])


    # mesh = marching_cubes(network_manager_root, hp['sampling_resolution_3d'])
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

    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1, 1)
    #ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.scatter(*xyzs[(labels != 1) & (labels != 0)].T, c=labels[(labels != 1) & (labels != 0)], alpha=0.2)
    ax.scatter(*xyzs.T, c=labels, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    main()

'''
todo
check if tree is helping or its just capacity 

read: https://lioryariv.github.io/volsdf/  https://lioryariv.github.io/idr/

loss: add grad*tangent = 0 in boundary
          eikonal in boundary
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

'''
def get_indices_to_flip(board, i, j):

    # convert board to binary metrix
    #for k in range(8):
    #    for l in range(8):
    #        board[k][l] = 1 if board[k][l] == 'black' else 0

    # convert i,l to list of its bin components for example 5 -> [1, 0, 1]
    # since i,j are in [0, 7] this results in an array of size 3
    i = list(map(int, list(bin(i))[2:]))
    j = list(map(int, list(bin(j))[2:]))

    # xoring board row-wise to a single bool array of size 8
    r = []
    for row in board:
        r.append( row[0] ^ row[1]^ row[2]^ row[3]^ row[4]^ row[5]^ row[6]^ row[7] )

    # same with columns
    c = []
    for colum in board.T:
        c.append( colum[0] ^ colum[1]^ colum[2]^ colum[3]^ colum[4]^ colum[5]^ colum[6]^ colum[7]  )

    def code(r):
        rtag = [None]*3
        rtag[0] = r[1]^        r[3]^       r[5]      ^ r[7]
        rtag[1] =       r[2]^  r[3]^             r[6]^ r[7]
        rtag[2] =                    r[4]^ r[5]^ r[6]^ r[7]
        return rtag


    # xor each bit of i (or j) with the code and convert to int
    itag = map(str, element_wise_xor(code(r), i))
    jtag = map(str, element_wise_xor(code(c), i))

    itag = int(''.join(itag), 2)
    jtag = int(''.join(jtag), 2)

    i =  int(''.join(map(str, code(c))), 2)
    j =  int(''.join(map(str, code(r))), 2)

    return itag, jtag, i, j


def element_wise_xor(a,b):
    return [i^j for i,j in zip(a,b)]


board = np.random.randint(0,2,(8,8))
i_tag,j_tag, i, j = get_indices_to_flip(board, 5,5)
board[i_tag, j_tag] = 1-board[i_tag, j_tag]
get_indices_to_flip(board, 5,5)
'''