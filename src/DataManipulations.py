__author__ = 'Sebastijan'

import os
import fnmatch
import copy

import cv2
import numpy as np


class DataCollector():
    def __init__(self, input_file):
        """
            Class containing description of the ground truth (landmarks) and functionalities regarding that

            file : file containing the landmarks

            Convention for points (vertical, horizontal) -- as mapping for OpenCV
        """
        self.points = []
        if input_file is not None:
            self._read_landmarks(input_file)
        self.centroid = None
        self._update_centroid(weights=None)
        self.scales = []
        self.scale_factor = None

    def _read_landmarks(self, input_file):
        """
            Method for parsing a landmark file
        """
        tmp = open(input_file).readlines()

        for ind in range(0, len(tmp), 2):
            self.points.append(np.array([float(tmp[ind + 1].strip()), float(tmp[ind].strip())]))

        self.points = np.array(self.points)

    def as_vector(self):
        """
            return points as [x_1,y_1, ..., x_n, y_n]
            x -- vertical value (height)
            y -- horizontal value (width)
        """

        return np.hstack(self.points)

    def as_matrix(self):
        """
            return points as matrix (height, width)
        """
        return self.points

    def read_vector(self, data_vector, weights=None):
        """
            Read vector of point and store it in self.points

            params:
                data_vector = vector of points in OpenCV mapping style [y_1, x_1, ..., y_n, x_n], numpy array
        """
        self.points = np.zeros((len(data_vector) / 2, 2))
        self.points[:, 0] = copy.copy(data_vector[range(0, len(data_vector), 2)])
        self.points[:, 1] = copy.copy(data_vector[range(1, len(data_vector), 2)])
        self._update_centroid(weights=weights)

    def read_points(self, points, weights=None):
        """
            Method reads point in a matrix format [[y_1, x_1], [y_2, x_2], ..., [y_n, x_n]]
        """
        self.points = copy.copy(points)
        self._update_centroid(weights=weights)

    def _update_centroid(self, weights=None):
        """
            Method updates the centroid
            -- used after translating the points' centroid to the origin
        """
        if weights is None:
            self.centroid = np.mean(self.points, axis=0)
        else:
            self.centroid = np.zeros((1, len(self.points[0])))
            for ind in range(len(self.points)):
                self.centroid += self.points[ind, :] * weights[ind]

    def translate_to_origin(self, weights=None):
        """
            Method translates the points so that centroid is in the origin [0, 0]

            params:
                weights : (1, num_points) numpy array of weights
                        -- if None, arithmetic mean is calculated
        """
        if weights is None:
            centroid = np.mean(self.points, axis=0)
        else:
            centroid = np.zeros((1, len(self.points[0])))
            for ind in range(len(weights)):
                centroid += self.points[ind, :] * weights[ind]
                #in case when sum of weights doesn't correspond to 1
            centroid = centroid.dot(1. / weights.sum())

        self.points = self.points - centroid

        #update_centroid
        self._update_centroid(weights=weights)

    def translate_to_reference(self, reference_centroid, weights=None):
        """
            Method translates the points for a given centroid

            params:
                reference_centroid : (1, num_dims) numpy array of centroid
        """

        self.points = self.points - reference_centroid
        self._update_centroid(weights=weights)

    def scale_to_unit(self):
        """
            Method scales each landmark point to the unit distance from the origin
        """
        self.scale_factor = self.points - self.centroid
        self.scale_factor = np.power(self.scale_factor, 2)
        self.scale_factor = self.scale_factor.dot(1. / len(self.points))
        self.scale_factor = self.scale_factor.sum()
        self.scale_factor = np.sqrt(self.scale_factor)

        self.points = self.points.dot(1. / self.scale_factor)
        self._update_centroid()

    def check_distance(self):

        result = self.points - self.centroid
        result = np.power(result, 2)
        result = result.dot(1. / len(self.points))
        result = result.sum()
        result = np.sqrt(result)

        return result

    def rescale(self):
        """
            Method rescales each landmark point to it's original distance
        """
        self.points = self.points.dot(self.scale_factor)

    def rescale_with_factor(self, factor):
        """
            Method rescales each landmark point with the given factor
        """
        self.points = self.points.dot(factor)

    def rotate(self, angle):
        """
            Method rotates each landmark point for a given angle
        """
        # rotation matrix, adapted to the (y, x) convention (height, width)
        rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        for ind in range(len(self.points)):
            self.points[ind, :] = self.points[ind, :].dot(rot_matrix)

    def __sub__(self, other):
        """
            Method for subtractions of two DataCollectors

            params:
                other : DataCollector

            returns:
                self.points - other.points
        """
        return self.points - other.points

    def realign_to_absolute(self):
        mins = np.min(self.points, axis=0)
        self.points = self.points - mins


class Plotter():
    def __init__(self):
        """
            Class implementing plotter functionality for visualization of landmark points
        """

    @staticmethod
    def render_landmarks(data_collector):
        """
            Method visualizes landmark points in a given data_collector of type DataCollector
        """

        points = data_collector.as_matrix()
        max_y = points[:, 0].max()
        min_y = points[:, 0].min()
        max_x = points[:, 1].max()
        min_x = points[:, 1].min()

        img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))

        for i in range(len(points)):
            img[int(points[i, 0] - min_y), int(points[i, 1] - min_x)] = 1

        cv2.imshow('Rendered shape', img)
        cv2.waitKey(0)

    @staticmethod
    def render_over_image(data_collector, img):
        """
            Method render a shape described with landmarks over the input image

            params:
                data_collector = landmark points as DataCollector class
                input_img = image to render on, loaded with OpenCV
        """

        points = data_collector.as_matrix()
        for i in range(len(points) - 1):
            #input_img[self.points[i, 0], self.points[i, 1], 1] = 255
            cv2.line(img, (int(points[i, 1]), int(points[i, 0])),
                     (int(points[i + 1, 1]), int(points[i + 1, 0])), (0, 255, 0))

        height = 500
        scale = height / float(img.shape[0])
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)

        cv2.namedWindow('rendered image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rendered image', window_width, window_height)
        cv2.imshow('rendered image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def display_image(img, window_name):
        """
            Method display the image so that it fits the screen
        """
        height = 500
        scale = height / float(img.shape[0])
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def render_multiple_landmarks(matrix):
        """
            Method plots multiple landmark points to the same image

            params:
                matrix : a matrix where each row represents a landmark point [y_1, x_1, ..., y_n, x_n]

            TODO : not fully functional yet, problems with positioning of each tooth
        """

        collector = DataCollector(None)

        # window initialization
        collector.read_vector(matrix[0, :])
        points = collector.as_matrix()
        max_y = points[:, 0].max()
        min_y = points[:, 0].min()
        max_x = points[:, 1].max()
        min_x = points[:, 1].min()

        img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))

        #plot each point
        for ind in range(len(matrix)):

            collector.read_vector(matrix[ind, :])
            points = collector.as_matrix()

            for i in range(len(points)):
                img[points[i, 0] - min_y, points[i, 1] - min_x] = 1

        cv2.imshow('Rendered shape', img)
        cv2.waitKey(0)


def collect_vectors(input_folder, tooth_number, dims):
    """
        Function that collects the models for tooth defined in teeth_numbers (can be multiple of them)

        params:
            input_folder : folder with landmarks files
            tooth_number : identifier of tooth
            dims : dimensionality of vectors (used for initialization of resulting matrix) -- 2*number_of_points
    """

    files = fnmatch.filter(os.listdir("{}/.".format(input_folder)), "*-{}.txt".format(str(tooth_number)))
    res_matrix = np.zeros((len(files), dims))

    for i in range(len(files)):
        tmp_obj = DataCollector("{}/{}".format(input_folder, files[i]))
        res_matrix[i, :] = tmp_obj.as_vector()

    return res_matrix
