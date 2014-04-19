__author__ = 'Sebastijan'

import cv2
from scipy.ndimage import morphology
import numpy as np
from matplotlib import pyplot as plt


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

    @staticmethod
    def calculate_fourier(img):
        """
            Method calculates thr Fourier coefficients of an 2-D numpy array
        """
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        return np.fft.fftshift(dft)

    @staticmethod
    def high_pass_filter(spectral_image, shape, h_offset=5, v_offset=5):
        """
            Method implements a high pass filter for fourier components
        """
        rows, cols = shape
        crow, ccol = rows/2, cols/2

        spectral_image[(crow-v_offset):(crow+v_offset), (ccol-h_offset):(ccol+h_offset)] = 0

        return spectral_image

    @staticmethod
    def low_pass_filter(spectral_image, shape, v_offset=100, h_offset=100):
        """
            Method implements a low pass filter for fourier components
        """
        rows, cols = shape
        crow, ccol = rows/2, cols/2

        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[(crow-v_offset):(crow+v_offset), (ccol-h_offset):(ccol+h_offset)] = 1

        return spectral_image * mask

    @staticmethod
    def inverse_fourier_transform(spectral_image):
        """
            Method computes the inverse Fourier transform of a given image
        """
        f_ishift = np.fft.ifftshift(spectral_image)
        img_back = cv2.idft(f_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    @staticmethod
    def display_fourier(spectrum):
        plt.imshow(spectrum, cmap='gray'), plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def to_magnitude(dft_shift):
        return 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
