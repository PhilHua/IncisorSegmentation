__author__ = 'Sebastijan'

import cv2
from scipy.ndimage import morphology
import numpy as np


class Preprocessor():

    def __init__(self):
        """
            Class contains operations for cleaning the image
        """

    @staticmethod
    def equalize_histogram(input_image):
        """
            Method equalizes the histogram of a given image

            params:
                input_image : cv2 loaded image
        """
        #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(input_image)

    @staticmethod
    def top_hat_transform(image):
        """
            Method calculates the top hat transformation of a given image
        """

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        structure = np.array([[1., 1., 2., 5., 2., 1.],
                              [1., 2., 5., 5., 5., 1.],
                              [1., 5., 5., 10., 5., 1.],
                              [1., 1., 5., 5., 5., 1.],
                              [1., 1., 2., 5., 2., 1.]])

        return morphology.white_tophat(image, size=400)

    @staticmethod
    def bottom_hat_transform(image):
        """
            Method calculates the bottom hat transformation of a given imaeg
        """

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        structure = np.array([[1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.]])
        return morphology.black_tophat(image, size=80)