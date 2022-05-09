import json

import pymeshlab


def hausdorff_distance(original_mesh_path, recon_mesh_path, save_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(original_mesh_path)
    ms.load_new_mesh(recon_mesh_path)
    res1 = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, sampleedge=True)
    res2 = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, sampleedge=True)
    print(f'original={original_mesh_path}, recon={recon_mesh_path}')
    print(f'\nhausdorff:\n{res1}\n{res2}\ndist={max(res1["max"], res2["max"])/ res1["diag_mesh_0"]}')
    if save_path is not None:
        with open(save_path, 'w') as fp:
            json.dump(res1, fp, indent=4)
            json.dump(res2, fp, indent=4)


if __name__ == '__main__':
    hausdorff_distance('./mesh/armadillo.obj', './mesh/eight.off', None)
