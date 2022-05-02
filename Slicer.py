import CSL
from meshcut import cross_section
import pywavefront


def make_csl():
    scene = pywavefront.Wavefront('./mesh/armadillo.obj', collect_faces=True)
    cross_section(scene.vertices, scene.meshes[0].faces, plane_orig=(1.2, -0.125, 0), plane_normal=(1, 0, 0))
    pass


if __name__ == '__main__':
    make_csl()