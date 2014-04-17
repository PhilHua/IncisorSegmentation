__author__ = 'Sebastijan'

import cv2


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
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(input_image)
