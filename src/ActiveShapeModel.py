__author__ = 'Sebastijan'

import numpy as np
import utils
import DataManipulations


class ReferentModel():

    def __init__(self, points):
        """
            Class implementing functionality for a referent model

            params:
                points : matrix with landmark points, each row represents one point [y_1, x_1,.., y_n, x_n] (OpenCV-style mapping)
        """
        self.points = points
        self.mean_shape = None

    def mean_model(self):
        """
            Method calculates mean model from self.points
        """
        return np.mean(self.points, axis=0)

    def _calculate_distances_to_points(self):
        """
            Method calculates the distance of each point to the neighbouring points (closed loop assumed)

            returns:
                distance matrix of dimensions (num_images, num_landmarks)
        """
        distances = np.zeros((len(self.points), len(self.points[0]) / 2))
        collector = DataManipulations.DataCollector(None)

        for img in range(len(self.points)):
            collector.read_vector(self.points[img, :])
            points = collector.as_matrix()

            for ref_point_ind in range(len(points)):
                distances[img, ref_point_ind] = sum([utils.euclidean_distance(points[ref_point_ind, :], x) for x in points])

        return distances

    def _calculate_weights(self, norm="normalize"):
        """
            Calculation of weight matrix for shape alignment
            -- weights are proportional to the standard deviation of a point

            returns:
                weights : (1, num_points) array of weights, weight[i] corresponds to landmark point i
        """
        distances = self._calculate_distances_to_points()
        weights = np.std(distances, axis=0, dtype=np.float64)
        weights = np.array([1. / x for x in weights])

        #normalize weights
        if norm == "normalize":
            # so they sum up to 1
            weights = weights / weights.sum()
        elif norm == "scale":
            # scaled to the interval [0,1]
            weights = weights / weights.max()

        return weights

    def _convert_matrix_to_collection(self):
        """
            Method converts the points given in matrix format to a list of DataCollector objects (easier to manipulate )
        """

        collection = []
        for ind in range(len(self.points)):
            tmp = DataManipulations.DataCollector(None)
            tmp.read_vector(self.points[ind, :])

            collection.append(tmp)

        self.points = collection

    def _convert_collection_to_matrix(self):
        """
            Method converts the points given as DataCollector objects to a matrix form
        """

        collection = []
        for item in self.points:
            collection.append(item.as_vector())

        self.points = np.array(collection)

    def _find_angle(self, current_shape):
        """
            Method finds the angle between reference shape in self.mean_shape and current_shape
        """

    def align(self):
        """
            Method implements the alignment of landmarks based on Generalized Procrustes analysis

            Notes:
                -- weighted mean is calculated when needed, not just arithmetic mean
        """
        weights = self._calculate_weights()
        self._convert_matrix_to_collection()

        #translating points to the origin to avoid need for translation
        for item in self.points:
            item.translate_to_origin()

        #scale points to unit distance
        for item in self.points:
            item.scale_to_unit()