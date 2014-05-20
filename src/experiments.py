__author__ = 'Sebastijan'

import cv2

from DataManipulations import Plotter, collect_vectors
from ActiveShapeModel import ReferentModel, VarianceModel, ActiveShape

res = collect_vectors('../data/Landmarks/original', '5', 80)

referent = ReferentModel(res)
referent.align()
referent.rescale_and_realign()

variance = VarianceModel(referent)
variance.obtain_components()

asm = ActiveShape(cv2.imread('../data/Radiographs/01.tif'), (500, 500), variance)
asm._calculate_normals()
Plotter.render_normals(asm)

