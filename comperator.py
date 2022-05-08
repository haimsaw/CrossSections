import json

import pymeshlab


def hausdorff_distance(original_mesh_path, recon_mesh_path, save_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(original_mesh_path)
    ms.load_new_mesh(recon_mesh_path)
    res = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, sampleedge=True)
    print(f'original={original_mesh_path}, recon={recon_mesh_path}')
    print(f'hausdorff_distance={res}')
    if save_path is not None:
        with open(save_path, 'w') as fp:
            json.dump(dict, fp, indent=4)
    return res


if __name__ == '__main__':
    hausdorff_distance('./mesh/armadillo.obj', './mesh/eight.off', None)
