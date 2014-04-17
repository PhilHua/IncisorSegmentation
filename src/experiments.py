__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel

res = DataManipulations.collect_vectors('../data/Landmarks/original', '4', 80)

referent = ActiveShapeModel.ReferentModel(res)
referent.align()
referent.rescale_and_realign()
print referent.points[0].points

variance = ActiveShapeModel.VarianceModel(referent)
