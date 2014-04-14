__author__ = 'Sebastijan'

import cv2
import numpy as np
import os
import fnmatch


class DataCollector():

    def __init__(self, input_file):
        """
            Class containing description of the ground truth (landmarks)

            file : file containing the landmarks

            Convention for points (vertical, horizontal) -- as mapping for OpenCV
        """
        self.points = []
        self._read_landmarks(input_file)

    def _read_landmarks(self, input_file):
        """
            Method for parsing a landmark file
        """
        tmp = open(input_file).readlines()

        for ind in range(0, len(tmp), 2):
            self.points.append(np.array([float(tmp[ind+1].strip()), float(tmp[ind].strip())]))

        self.points = np.array(self.points)

    def render_shape(self):
        """
            Method displays read landmarks
        """
        max_y = self.points[:, 0].max()
        min_y = self.points[:, 0].min()
        max_x = self.points[:, 1].max()
        min_x = self.points[:, 1].min()

        img = np.zeros((int((max_y - min_y)*1.1), int((max_x - min_x)*1.1)))

        for i in range(len(self.points)):
            img[self.points[i, 0] - min_y, self.points[i, 1] - min_x] = 1

        cv2.imshow('image', img)
        cv2.waitKey(0)

    def render_over_image(self, input_img):
        """
            Method render a shape described with landmarks over the input image

            params:

                input_img = image to render on, loaded with OpenCV
        """

        for i in range(len(self.points) - 1):
            #input_img[self.points[i, 0], self.points[i, 1], 1] = 255
            cv2.line(input_img, (int(self.points[i, 1]), int(self.points[i, 0])),
                     (int(self.points[i+1, 1]),  int(self.points[i+1, 0])), (0, 255, 0))

        height = 500
        scale = height / float(input_img.shape[0])
        window_width = int(input_img.shape[1] * scale)
        window_height = int(input_img.shape[0] * scale)

        cv2.namedWindow('rendered image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rendered image', window_width, window_height)
        cv2.imshow('rendered image', input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def as_vector(self):
        """
            return points as [x_1,y_1, ..., x_n, y_n]
            x -- vertical value (height)
            y -- horizontal value (width)
        """

        return np.hstack(self.points)

    def as_matrix(self):
        """
            return points as matrix
        """
        return self.points


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

