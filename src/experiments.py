__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel


res = DataManipulations.collect_vectors('../data/Landmarks/original', '1', 80)

referent = ActiveShapeModel.ReferentModel(res)
referent.align()
referent.rescale_and_realign()

variance = ActiveShapeModel.VarianceModel(referent)
variance.obtain_components()

print variance.get_eigenvalues()

