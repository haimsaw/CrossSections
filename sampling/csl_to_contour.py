import numpy as np
from simplification.cutil import simplify_coords


def csl_to_contour(csl, save_path):
    # format: https://www.cse.wustl.edu/~taoju/lliu/paper/ctr2suf/program.html
    file_name = f'{save_path}{csl.model_name}.contour'

    with open(file_name, 'w') as f:
        f.write(f'{len(csl.planes)}\n')
        for plane in csl.planes:
            f.write(f'{plane.plane_params[0]:.10f} {plane.plane_params[1]:.10f} {plane.plane_params[2]:.10f} {-1*plane.plane_params[3]:.10f}\n')

            verts, edges = plane.vertices, plane.edges

            f.write(f'{len(verts)} {len(edges)}\n')
            for v in verts:
                f.write('{:.10f} {:.10f} {:.10f}\n'.format(*v))
            for e in edges:
                # todo this only deals with single label csl
                f.write('{} {} 1 0\n'.format(*e))
