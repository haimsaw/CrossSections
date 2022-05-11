import multiprocessing
import os
from multiprocessing import Pool, cpu_count

from hp import get_csl, HP
from ChainTrainer import ChainTrainer
from SlicesDataset import SlicesDataset
from Renderer import *
from Mesher import *
from Comperator import hausdorff_distance


def train_cycle(csl, hp, trainer, should_calc_density, save_path):
    slices_dataset = SlicesDataset.from_csl(csl, sampling_resolution=hp.root_sampling_resolution_2d,
                                            sampling_margin=hp.sampling_margin, should_calc_density=should_calc_density)
    contour_dataset = None  # ContourDataset(csl, hp.n_samples_per_edge)

    trainer.prepare_for_training(slices_dataset, contour_dataset)

    with Pool(processes=cpu_count()) as pool:
        for i, epochs in enumerate(hp.epochs_batches):
            print(f'{"="*10} epochs batch {i}/{len(hp.epochs_batches)}:')
            new_cells, promise = trainer.get_refined_cells(pool)
            trainer.train_epochs_batch(epochs)
            try:
                handle_meshes(trainer, hp, save_path, i)
                pass
            except Exception as e:
                print(e)
            save_heatmaps(trainer, save_path, i)
            print('waiting for cell density calculation...')
            promise.wait()
            print('done waiting')
            trainer.update_data_loaders(new_cells)

    trainer.show_train_losses(save_path)


def handle_meshes(trainer, hp, save_path, label):
    #mesh_mc = marching_cubes(trainer, hp.sampling_resolution_3d)
    #mesh_mc.save(save_path + f'mesh_l{0}_mc.stl')

    #mesh_dc = dual_contouring(trainer, hp.sampling_resolution_3d, use_grads=True)
    #mesh_dc.save(save_path + f'mesh_dc_grad.obj')

    mesh_dc_no_grad = dual_contouring(trainer, hp.sampling_resolution_3d, use_grads=False)
    mesh_dc_no_grad.save(save_path + f'mesh{label}_dc_no_grad.obj')

    hausdorff_distance(f'{save_path}/original_mesh.stl', save_path + f'mesh{label}_dc_no_grad.obj',
                       f'{save_path}/hausdorff_distance{label}.json')

    '''
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


def main():
        hp = HP()
        save_path = f'./artifacts/test/'

        print(f'{"=" * 50} {save_path}')
        os.makedirs(save_path, exist_ok=True)

        csl = get_csl(hp.bounding_planes_margin, save_path)
        should_calc_density = hp.density_lambda > 0

        '''
        r = Renderer3D()
        r.add_scene(csl)
        r.show()
        '''

        trainer = ChainTrainer(csl, hp)

        with open(save_path + 'hyperparams.json', 'w') as f:
            f.write(hp.to_json())

        print(f'csl={csl.model_name}')

        train_cycle(csl, hp, trainer, should_calc_density, save_path)

        save_heatmaps(trainer, save_path, 'last')
        mesh_dc = handle_meshes(trainer, hp, save_path, 'last')

        renderer = Renderer3D()
        renderer.add_scene(csl)
        renderer.add_mesh(mesh_dc)
        # renderer.show()
        print(f'DONE {"=" * 50} {save_path}\n\n')


if __name__ == "__main__":
    main()

'''
todo
batch size to hp and invrese to 4048, icrese step size??
PE not bad
slicer https://shapely.readthedocs.io/en/stable/manual.html
sample - samples around edges + blue noise

start paper
compare with basic nerf and Robust optimization for topological surface reconstruction
talk with guy or amir hertz about visualizations
dual contouring - play with setting & debug

brain - increase persition of sampling


read: 
https://arxiv.org/abs/2202.01999 - nural dc
https://arxiv.org/pdf/2104.02699.pdf

create slicer for chamfer compare
serialize a tree (in case collab crashes)
increase sampling (in prev work he used 2d= 216, 3d=300)


point 2 mesh
'''
