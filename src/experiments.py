__author__ = 'Sebastijan'

import cv2

from Preprocess import Preprocessor
from DataManipulations import Plotter


img = cv2.imread('../data/Radiographs/01.tif')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_top = Preprocessor.top_hat_transform(img)
img_bottom = Preprocessor.bottom_hat_transform(img)

Plotter.display_image(img, 'Original image')
Plotter.display_image(img_top, 'Top hat filtered')
Plotter.display_image(img_bottom, 'Bottom hat filtered')
img = cv2.add(img, img_top)
img = cv2.subtract(img, img_bottom)

Plotter.display_image(img, 'Result')