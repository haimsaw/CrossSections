import os

from SlicesDataset import SlicesDataset
from Renderer import *
from Mesher import *
from OctnetTree import *
from hp import get_csl, HP


def train_cycle(csl, hp, tree, should_calc_density, save_path):
    d2_res = 2 ** (tree.depth + 1) * hp.root_sampling_resolution_2d
    slices_dataset = SlicesDataset(csl, sampling_resolution=d2_res, sampling_margin=hp.sampling_margin, should_calc_density=should_calc_density)
    contour_dataset = ContourDataset(csl, round(len(slices_dataset) / len(csl)))  # todo haim - remove param or fix this somehow
    print(f'slices={len(slices_dataset)}, contour={len(contour_dataset)}, samples_per_edge={round(len(slices_dataset) / len(csl))}')

    tree.prepare_for_training(slices_dataset, contour_dataset, hp)
    tree.train_network(epochs=hp.epochs)
    tree.show_train_losses(save_path)


def handle_meshes(tree, hp, save_path):
    mesh_mc = marching_cubes(tree, hp.sampling_resolution_3d, use_sigmoid=hp.sigmoid_on_inference)
    mesh_mc.save(save_path + f'mesh_l{tree.depth}_mc.stl')

    mesh_dc = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=True, use_sigmoid=hp.sigmoid_on_inference)
    mesh_dc.save(save_path + f'mesh_l{tree.depth}_dc_grad.obj')

    mesh_dc_no_grad = dual_contouring(tree, hp.sampling_resolution_3d, use_grads=False, use_sigmoid=hp.sigmoid_on_inference)
    mesh_dc_no_grad.save(save_path + f'mesh_l{tree.depth}_dc_no_grad.obj')

    return mesh_dc


def save_heatmaps(tree, save_path, hp):
    heatmap_path = save_path + f'/heatmaps_l{tree.depth}/'

    os.makedirs(heatmap_path, exist_ok=True)

    for dim in (0, 1, 2):
        for dist in np.linspace(-0.5, 0.5, 3):
            renderer = Renderer2D()
            renderer.heatmap([100] * 2, tree, dim, dist, True, hp.sigmoid_on_inference)
            renderer.save(heatmap_path)
            renderer.clear()


def main():
    save_path = './artifacts/'
    hp = HP()
    csl = get_csl(hp.bounding_planes_margin)
    should_calc_density = hp.initial_density_lambda > 0 or hp.inter_lambda > 0
    tree = OctnetTree(csl, hp.oct_overlap_margin, hp.hidden_layers,
                      get_embedder(hp.num_embedding_freqs, hp.spherical_coordinates))

    with open(save_path + 'hyperparams.json', 'w') as f:
        f.write(hp.to_json())

    print(f'csl={csl.model_name}')
    print(f'loss: density={hp.initial_density_lambda}, eikonal={hp.eikonal_lambda}, contour_val={hp.contour_val_lambda},'
          f' contour_normal={hp.contour_normal_lambda}, contour_tangent={hp.contour_tangent_lambda} '
          f' inter={hp.inter_lambda} off={hp.off_surface_lambda}')

    for _ in range(hp.depth):
        train_cycle(csl, hp, tree, should_calc_density, save_path)
        save_heatmaps(tree, save_path, hp)
        mesh_dc = handle_meshes(tree, hp, save_path)

        renderer = Renderer3D()
        renderer.add_scene(csl)
        renderer.add_mesh(mesh_dc)
        renderer.show()


if __name__ == "__main__":
    main()

'''
todo
check if tree is helping or its just capacity 
delete INetManager

read: 
https://arxiv.org/abs/2202.01999 - nural dc
https://arxiv.org/pdf/1807.02811.pdf - Bayesian Optimization

create slicer for chamfer compare
track experiments with mlflow?
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
