import json
import os
from multiprocessing import Pool, cpu_count

from sampling.Slicer import make_csl_from_mesh
from sampling.csl_to_contour import csl_to_contour
from sampling.csl_to_point2mesh import csl_to_point2mesh
from hp import HP, args
from ChainTrainer import ChainTrainer
from sampling.SlicesDataset import SlicesDataset
from Renderer import *
from Mesher import *
from Comperator import hausdorff_distance
from time import time
from sampling.CSL import CSL
import pickle


def train_cycle(csl, hp, trainer, save_path, model_name):
    total_time = 0
    # with Pool(processes=cpu_count()//2) as pool:
    data_sets = []

    ts = time()
    trainer.prepare_for_training()
    te = time()

    total_time += te - ts
    for i, epochs in enumerate(hp.epochs_batches):
        print(f'\n\n{"="*10} epochs batch {i+1}/{len(hp.epochs_batches)}:')

        data_sets.append(SlicesDataset.from_csl(csl, pool=None, hp=hp, gen=i))
        data_sets[-1].to_ply(save_path + f"datast_gen_{i}.ply")
        trainer.update_data_loaders(data_sets)

        ts = time()
        trainer.train_epochs_batch(epochs)
        te = time()
        total_time += te - ts

        trainer.save_to_disk(save_path+f"trained_model_{i}.pt")
        trainer.show_train_losses(save_path)

        try:
            print('meshing')
            handle_meshes(trainer, hp.intermediate_sampling_resolution_3d, save_path, i, model_name)
            pass
        except Exception as e:
            print(e)

        # print('heatmaps')
        # save_heatmaps(trainer, save_path, i)

    print(f'\n\n done train_cycle time = {total_time} sec')


def handle_meshes(trainer, sampling_resolution_3d, save_path, label, name):
    #mesh_mc = marching_cubes(trainer, hp.sampling_resolution_3d)
    #mesh_mc.save(save_path + f'mesh_l{0}_mc.stl')

    #mesh_dc = dual_contouring(trainer, hp.sampling_resolution_3d, use_grads=True)
    #mesh_dc.save(save_path + f'mesh_dc_grad.obj')

    ts = time()
    mesh_dc_no_grad = dual_contouring(trainer, sampling_resolution_3d, use_grads=False)
    te = time()

    print(f'meshing time of {label}= {te - ts} sec')

    mesh_dc_no_grad.save(save_path + f'mesh{label}_dc_no_grad.obj')

    '''
    try:
        hausdorff_distance(f"data/csl_from_mesh/{name}_scaled.stl", save_path + f'mesh{label}_dc_no_grad.obj',
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


def main(model_name):
        hp = HP()
        save_path = f'{args.out_dir}/{model_name}/'
        os.makedirs(save_path, exist_ok=True)

        print(f'{"=" * 50} {save_path}')

        csl = CSL.from_csl_file(f"./data/csl-files/{model_name}.csl")
        # csl = CSL.from_csl_file(f"./data/csl_from_mesh/{name}_from_mesh.csl")

        csl.adjust_csl(args.bounding_planes_margin)

        print(f'csl={csl.model_name} slices={len([p for p in csl.planes if not p.is_empty])}, n edges={len(csl)}')

        trainer = ChainTrainer(csl, hp)

        with open(save_path + 'hyperparams.json', 'w') as f:
            f.write(json.dumps(hp, default=lambda o: o.__dict__, sort_keys=True, indent=4))
            f.write(json.dumps(args, default=lambda o: o.__dict__, sort_keys=True, indent=4))

        train_cycle(csl, hp, trainer, save_path, model_name)

        mesh_dc = handle_meshes(trainer, hp.sampling_resolution_3d, save_path, 'last', model_name)

        save_heatmaps(trainer, save_path, 'last')

        print(f'DONE {"=" * 50} {save_path}\n\n')


if __name__ == "__main__":

    for model_name in ['Heart-25-even-better', 'Vetebrae', 'Skull-20', 'Brain']:
    # for model_name in ['armadillo', 'lamp004_fixed', 'eight_15', 'eight_20']:
        try:
            main(model_name)
        except Exception as e:
            print(f"an error has occurred, continuing: {e}")

        continue


        hp = HP()
        csl = make_csl_from_mesh(f'./data/obj/{model_name}.obj', './data/csl_from_mesh/')
        # csl = CSL.from_csl_file(f"./data/csl_from_mesh/{model_name}_from_mesh.csl")

        print(f'csl_len = {len(csl)}')
        csl.adjust_csl(args.bounding_planes_margin)
        '''csl.planes = csl.planes[15:16] + csl.planes[21:22]

        csl.planes[0].connected_components = csl.planes[0].connected_components[0:1]
        csl.planes[0].vertices = csl.planes[0].vertices[:sum(map(len, csl.planes[0].connected_components))]

        csl.planes[1].connected_components = csl.planes[1].connected_components[0::3]
        vs = csl.planes[1].vertices
        csl.planes[1].vertices = np.empty((0, 3))
        for cc in csl.planes[1].connected_components:
            start = len(csl.planes[1].vertices)
            csl.planes[1].vertices = np.append(csl.planes[1].vertices, vs[cc.vertices_indices], axis=0)
            cc.vertices_indices = np.array(range(start, len(csl.planes[1].vertices)))
        '''
        csl_to_contour(csl, "./data/for_CycleGrouping/")




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


    '''todo:
    1. refine
    2. petrube points proportional to to dist
    3. run meny examples'''