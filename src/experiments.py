__author__ = 'Sebastijan'

import cv2

from Preprocess import Preprocessor
from DataManipulations import Plotter


img = cv2.imread('../data/Radiographs/01.tif', 0)
Plotter.display_image(img, 'Original image')

dft_shift = Preprocessor.calculate_fourier(img)

dft_shift = Preprocessor.high_pass_filter(dft_shift, img.shape)

dft_shift = Preprocessor.low_pass_filter(dft_shift, img.shape)
#
img_back = Preprocessor.inverse_fourier_transform(dft_shift)
Preprocessor.display_fourier(img_back)
