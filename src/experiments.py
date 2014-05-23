__author__ = 'Sebastijan'

import cv2
import numpy
from matplotlib import pyplot as plt

from DataManipulations import Plotter, collect_vectors
from ActiveShapeModel import ReferentModel, VarianceModel, ActiveShape
from Preprocess import Preprocessor


img_ori = cv2.imread('../data/Radiographs/01.tif')
img = cv2.imread('../data/Radiographs/01.tif', 0)
img = cv2.bilateralFilter(img, 11, 75, 75)

img_top = Preprocessor.top_hat_transform(img)
img_bottom = Preprocessor.bottom_hat_transform(img)
img = cv2.add(img, img_top)
img = cv2.subtract(img, img_bottom)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
result = numpy.sqrt(sobelx**2 + sobely**2)
#result = cv2.imread('edges_temporary.png', 0)
plt.imshow(result, 'gray')
#plt.imsave('edges_temporary.png', result)
plt.show()

#sobelx = sobelx.dot(1./(sobelx.max()))*255.

#plt.imshow(sobelx, 'gray')
#plt.show()


res = collect_vectors('../data/Landmarks/original', '1', 80)
referent = ReferentModel(res)
referent.align()
referent.rescale_and_realign()

variance = VarianceModel(referent)
variance.obtain_components()
asm = ActiveShape(result, (867, 1364), variance)
asm.update_shape()
#asm.plot(img_ori)
Plotter.render_normals(asm)