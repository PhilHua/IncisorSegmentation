__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel

res = DataManipulations.collect_vectors('../data/Landmarks/original', '4', 80)

referent = ActiveShapeModel.ReferentModel(res)
referent.align()

print referent.retrieve_as_matrix()