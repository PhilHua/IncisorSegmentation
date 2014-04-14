__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel

res = DataManipulations.collect_vectors('../data/Landmarks/original', '1', 80)

referent = ActiveShapeModel.ReferentModel(res)
weights = referent.align()

