import numpy as np


class Cell:
    def __init__(self, pixel_center, pixel_radius, labeler, xyz_transformer, plane_id, generation=0):
        #assert pixel_radius > 0
        self._label = None

        self.pixel_center = pixel_center
        self.pixel_radius = pixel_radius

        self.labeler = labeler
        self.xyz_transformer = xyz_transformer
        self.xyz = xyz_transformer(np.array([self.pixel_center]))[0]
        self.plane_id = plane_id
        self.generation = generation

    @property
    def density(self):
        if self._label is None:
            self._label = self._get_label()
        return self._label

    @property
    def boundary(self):
        return np.array([[1, 1],
                        [1, -1],
                        [-1, -1],
                        [-1, 1]]) * self.pixel_radius + self.pixel_center

    '''
    using Hoeffding's inequality we get that for the result to be in range of +- eps=0.1 from actual value
    w.p 1-alpha=1-0.001=99.9%
    we need 380=math.log(2/alpha)/(2*eps*eps) samples 
    '''
    def _get_label(self, accuracy=400):
        rnd = np.random.random_sample((accuracy, 2))
        sampels = self.pixel_radius * (2 * rnd - 1) + self.pixel_center
        labels = self.labeler(sampels)
        return sum(labels) / accuracy

    def split_cell(self):
        new_cell_radius = self.pixel_radius / 2
        new_centers = np.array([[1, 1],
                                [1, -1],
                                [-1, 1],
                                [-1, -1]]) * new_cell_radius + self.pixel_center

        # its ok to use self.labeler and self.xyz_transformer since the new cells are on the same plane
        return [Cell(xy, new_cell_radius, self.labeler, self.xyz_transformer, self.plane_id, self.generation+1) for xy in new_centers]
