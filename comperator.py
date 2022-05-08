import json

import pymeshlab


def hausdorff_distance(mesh_path_0, mesh_path_1, save_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path_0)
    ms.load_new_mesh(mesh_path_1)
    res = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, sampleedge=True)
    print(res)
    if save_path is not None:
        with open(save_path, 'w') as fp:
            json.dump(dict, fp, indent=4)


hausdorff_distance('./mesh/armadillo.obj', './mesh/eight.off', None)
