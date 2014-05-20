__author__ = 'Sebastijan'

import copy
import types

import numpy as np
from sklearn.decomposition import PCA

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

        #if type(self.points) is type([]):
        if isinstance(self.points, types.ListType):
            coll = self._convert_collection_to_matrix()
            return np.mean(coll, axis=0)
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

        return np.array(collection)

    def align(self):
        """
            Method implements the alignment of landmarks based on Generalized Procrustes analysis

            Notes:
                -- weighted mean is calculated when needed, not just arithmetic mean
        """
        #weights = self._calculate_weights()
        self._convert_matrix_to_collection()

        #translating points to the origin to avoid need for translation
        for item in self.points:
            item.translate_to_origin()

        #scaling shape to unit distance
        for item in self.points:
            item.scale_to_unit()
            #print item.scale_factor

        self.mean_shape = copy.copy(self.points[0])
        old_mean_shape = DataManipulations.DataCollector(None)
        old_mean_shape.read_points(np.zeros_like(self.mean_shape.points))

        while utils.is_converged(old_mean_shape, self.mean_shape) is not True:
            old_mean_shape.read_points(self.mean_shape.points)

            #recalculate new mean
            #self._convert_collection_to_matrix()
            self.mean_shape.read_vector(self.mean_model())
            #self._convert_matrix_to_collection()

            #normalize a new mean shape
            self.mean_shape.translate_to_origin()
            self.mean_shape.scale_to_unit()

            #realign each shape with new mean
            for item in self.points:
                angle = utils.rotation_alignment(self.mean_shape, item)
                item.rotate(angle)

                if item.check_distance() != 0.1:
                    item.translate_to_origin()
                    #item.scale_to_unit()
                    #print item.check_distance()

        #calculate mean scale factor
        scale_factor = 0.
        for item in self.points:
            scale_factor += item.scale_factor
        self.mean_shape.scale_factor = scale_factor / float(len(self.points))

    def retrieve_mean_model(self):
        """
            Get method for mean model

            returns:
                mean model as a DataCollector
        """
        return self.mean_shape

    def retrieve_as_matrix(self):
        """
            Method returns aligned dataset in a n_shapes x n_landmarks matrix
        """
        return self._convert_collection_to_matrix()

    def rescale_and_realign(self):
        """
            Method rescales every shape (after alignment each shape is unit) and translate landmarks from relative to absolute
                        position
        """

        for item in self.points:
            item.rescale_with_factor(self.mean_shape.scale_factor)
            item.realign_to_absolute()

        #realign the mean model also
        self.mean_shape.rescale()
        self.mean_shape.realign_to_absolute()


class VarianceModel():

    def __init__(self, ref_model):
        """
            Class containing the functionalities for examining the variance of shape

            params:
                ref_model : instance of ReferentModel, should be aligned, rescaled and realigned to an absolute position before
        """
        self.deviation = ref_model.retrieve_as_matrix()
        self.covariance = None
        self.pca_fitter = None
        self.mean_model = ref_model.retrieve_mean_model()

        for ind in range(len(self.deviation)):
            self.deviation[ind, :] = self.deviation[ind, :] - ref_model.mean_model()

    def _covariance_matrix(self):
        """
            Method calculates the covariance matrix
        """
        self.covariance = np.cov(self.deviation, rowvar=0)

    def obtain_components(self, num_comp=3):
        """
            Method calculates the principal components of the covariance matrix

            @params:
                num_comp: number of components to obtain

        """
        if self.covariance is None:
            self._covariance_matrix()

        self.pca_fitter = PCA(n_components=num_comp)
        self.pca_fitter.fit(self.covariance)

    def get_components(self):
        """
            Method returns the principal components calculated wth self.obtain_components
        """

        if self.pca_fitter is None:
            raise ValueError("Components not calculated")

        return self.pca_fitter.components_

    def get_variances_explained(self):
        """
            Method returns the variance ratios of calculated components
        """

        if self.pca_fitter is None:
            raise ValueError("Principal components not calculated")

        return self.pca_fitter.explained_variance_ratio_

    def get_eigenvalues(self):
        """
            Method returns the eigenvalues of the covariance matrix
        """

        eigenvals = np.linalg.eigvalsh(self.covariance)
        return sorted(eigenvals, reverse=True)[:len(self.pca_fitter.components_)]


class ActiveShape():

    def __init__(self, image, init_point, variance_model):
        """
            Class implements fitting procedure of Active shape models

            @params:
                image : image to fit the model in, openCV object
                init_point : initial centroid for the mean model
                variance_model : an instance of VarianceModel, with mean shape and principal components
        """
        self.image = image
        self.init_point = init_point
        self.current_shape = variance_model.mean_model
        self.p_components = variance_model.get_components()
        self.normals = []

    def _calculate_normals(self):
        """
            The method calculates the normals in each point. The normal is calculated as an average between
        """

        for ind in range(len(self.current_shape.points)):
            n1 = utils.normal(self.current_shape.points[ind-1, :], self.current_shape.points[ind, :])
            n2 = utils.normal(self.current_shape.points[ind, :], self.current_shape.points[(ind+1) % 40, :])

            self.normals.append(n1 + n2 / 2)

    def plot(self):
        """
            Renders the current mean shape over an image
        """

        points = self.current_shape.as_matrix()