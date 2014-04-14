__author__ = 'Sebastijan'

import numpy as np


class ReferentModel():

    def __init__(self, points):
        """
            Class implementing functionality for a referent model

            params:
                points : matrix with landmark points, each row represents one point [y_1, x_1,.., y_n, x_n] (OpenCV-style mapping)
        """
        self.points = points

    def mean_model(self):
        """
            Method calculates mean model from self.points
        """
        return np.mean(self.points, axis=0)

    def align(self):
        """
            Method implements the alignment of landmarks based on ???
        """
        pass