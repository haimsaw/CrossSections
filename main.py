import json
import os
from multiprocessing import Pool, cpu_count

from sampling.Slicer import make_csl_from_mesh
from sampling.csl_to_contour import csl_to_contour
from sampling.csl_to_point2mesh import csl_to_point2mesh
from hp import *
from ChainTrainer import ChainTrainer
from sampling.SlicesDataset import SlicesDataset
from Renderer import *
from Mesher import *
from Comperator import hausdorff_distance
from time import time
from sampling.CSL import CSL
import pickle
import random
import numpy as np
import torch
import traceback


def train_cycle(csl, model_name, hp, trainer, save_path, stats):
    total_time = 0

    data_sets = []

    trainer.prepare_for_training()

    for i, epochs in enumerate(hp.epochs_batches):
        print(f'\n\n{"=" * 10} epochs batch {i + 1}/{len(hp.epochs_batches)}:')

        ts = time()
        if args.no_refine and len(data_sets) == 0:
            data_sets = [SlicesDataset.from_csl(csl, hp=hp, gen=j) for j in range(len(hp.epochs_batches))]
        else:
            data_sets.append(SlicesDataset.from_csl(csl, hp=hp, gen=i))
        trainer.update_data_loaders(data_sets)
        stats['rasterize'].append(time() - ts)

        data_sets[i].to_ply(save_path + f"datast_gen_{i}.ply")

        ts = time()
        trainer.train_epochs_batch(epochs)
        stats['train'].append(time() - ts)

        stats['dataset_size'].append(len(data_sets[-1]))

        trainer.save_to_disk(save_path + f"trained_model_{i}.pt")
        trainer.show_train_losses(save_path)

        try:
            handle_meshes(model_name, trainer, hp.intermediate_sampling_resolution_3d, save_path, i, stats)
            pass
        except Exception as e:
            print(e)

        # print('heatmaps')
        # save_heatmaps(trainer, save_path, i)

    print(f'\n\n done train_cycle time = {total_time} sec')


def handle_meshes(model_name, trainer, sampling_resolution_3d, save_path, label, stats):
    # mesh_mc = marching_cubes(trainer, hp.sampling_resolution_3d)
    # mesh_mc.save(save_path + f'mesh_l{0}_mc.stl')

    # mesh_dc = dual_contouring(trainer, hp.sampling_resolution_3d, use_grads=True)
    # mesh_dc.save(save_path + f'mesh_dc_grad.obj')

    ts = time()
    mesh_dc_no_grad = dual_contouring(trainer, sampling_resolution_3d, use_grads=False)
    te = time()
    stats['meshing'][label] = te - ts

    print(f'meshing time of {label}= {te - ts} sec')

    mesh_dc_no_grad.save(save_path + f'mesh{label}_dc_no_grad.obj')

    '''
    try:
        hausdorff_distance(f"data/csl_from_mesh/{model_name}_scaled.stl", save_path + f'mesh{label}_dc_no_grad.obj',
                           f'{save_path}/hausdorff_distance{label}.json')
    except BaseException as e:
        print(f"unable to calc hausdorff_distance: {e}")

    for loop in [-1, -2, 5, 1]:
        mesh_dc_no_grad = dual_contouring(trainer, hp.sampling_resolution_3d, use_grads=False, loop=loop)
        mesh_dc_no_grad.save(save_path + f'mesh_loop{loop}_dc_no_grad.obj')
    '''

    return mesh_dc_no_grad


def save_heatmaps(trainer, save_path, label):
    heatmap_path = save_path + f'/heatmaps_{label}/'

    os.makedirs(heatmap_path, exist_ok=True)

    for dim in (0, 1, 2):
        for dist in np.linspace(-0.5, 0.5, 3):
            renderer = Renderer2D()
            renderer.heatmap([100] * 2, trainer, dim, dist, True)
            renderer.save(heatmap_path)
            renderer.clear()


def main(model_name, stats, save_path):

    hp = HP()
    os.makedirs(save_path, exist_ok=True)

    print(f'{"=" * 50} {save_path}')

    ts = time()
    if args.run_mri:
        csl = CSL.from_csl_file(f"./data/csl-files/{model_name}.csl")
    else:
        csl = CSL.from_csl_file(f"./data/csl_from_mesh/{model_name}_from_mesh.csl")

    csl.adjust_csl(args.bounding_planes_margin)
    stats['load_data'] = time() - ts

    stats['n_slices'] = len([p for p in csl.planes if not p.is_empty])
    stats['n_edges'] = len(csl)
    print(f'csl={csl.model_name} slices={stats["n_slices"]}, n edges={stats["n_edges"]}')

    trainer = ChainTrainer(csl, hp)

    with open(save_path + 'hyperparams.json', 'w') as f:
        f.write(json.dumps(hp, default=lambda o: o.__dict__, indent=4))
        f.write(json.dumps(args, default=lambda o: o.__dict__, indent=4))

    train_cycle(csl, model_name, hp, trainer, save_path, stats)

    with open(save_path + 'losses.json', 'w') as f:
        f.write(json.dumps(trainer.train_losses, default=lambda o: o.__dict__, indent=4))

    mesh_dc = handle_meshes(model_name, trainer, hp.sampling_resolution_3d, save_path, 'last', stats)

    save_heatmaps(trainer, save_path, 'last')

    stats['total_rastarization'] = sum(stats['rasterize'])
    stats['total_train'] = sum(stats['train'])
    stats['total_time'] = stats['total_train'] + stats['total_rastarization'] + stats['meshing']['last']


if __name__ == "__main__":

    if args.run_mri:
        models = mri_models
    else:
        models = from_mesh_models if args.model_name == 'all' else [args.model_name]

    errored = []

    for model_name in models:

        if args.seed > 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        save_path = f'{args.out_dir}/{model_name}/'

        stats = {'name': model_name,
                 'rasterize': [],
                 'train': [],
                 'dataset_size': [],
                 'meshing': {}}
        try:
            main(model_name, stats, save_path)

        except Exception as e:
            print('X' * 50)
            print(f"an error has occurred, continuing: {e}")
            traceback.print_exc()
            print(save_path)
            print('X' * 50)
            errored.append(model_name)

        finally:
            stats_str = json.dumps(stats, indent=4)
            with open(save_path + 'stats.json', 'w') as f:
                f.write(stats_str)

            print(stats_str)
            print(f'DONE {"=" * 70}\n{save_path}\n\n')

    print(f'errord={errored}')

    '''
    hp = HP()
    csl = make_csl_from_mesh(f'./data/obj/{model_name}.obj', './data/csl_from_mesh/')
    # csl = CSL.from_csl_file(f"./data/csl_from_mesh/{model_name}_from_mesh.csl")

    print(f'csl_len = {len(csl)}')
    csl.adjust_csl(args.bounding_planes_margin)
    csl.planes = csl.planes[15:16] + csl.planes[21:22]

    csl.planes[0].connected_components = csl.planes[0].connected_components[0:1]
    csl.planes[0].vertices = csl.planes[0].vertices[:sum(map(len, csl.planes[0].connected_components))]

    csl.planes[1].connected_components = csl.planes[1].connected_components[0::3]
    vs = csl.planes[1].vertices
    csl.planes[1].vertices = np.empty((0, 3))
    for cc in csl.planes[1].connected_components:
        start = len(csl.planes[1].vertices)
        csl.planes[1].vertices = np.append(csl.planes[1].vertices, vs[cc.vertices_indices], axis=0)
        cc.vertices_indices = np.array(range(start, len(csl.planes[1].vertices)))
   
    csl_to_contour(csl, "./data/for_CycleGrouping/")
     '''

    '''

    hp = HP()
    csl = CSL.from_csl_file("data/csl-files/Heart-25-even-better.csl")
    trainer = ChainTrainer(csl, hp)
    trainer.load_from_disk("./artifacts/hart_model_5.pt")

    scene_edges, scene_verts = csl.edges_verts

    samplig_res_3d = (50, 50, 50)
    xyzs = get_xyzs_in_octant(None, samplig_res_3d)
    labels = trainer.hard_predict(xyzs)

    ps.init()

    ps.look_at_dir((-2.5, 0.,0), (0., 0., 0.), (0,0,1))
    ps.set_ground_plane_mode("none")
    ps.set_screenshot_extension(".png")

    model = ps.register_point_cloud("model", xyzs[labels], material="candy", transparency=0.4)
    scene = ps.register_curve_network(f"scene", scene_verts, scene_edges, material="candy")

    scene.set_radius(scene.get_radius() / 1.5)
    # model.set_radius(model.get_radius() * 1)

    # ps.show()
    ps.screenshot()'''
