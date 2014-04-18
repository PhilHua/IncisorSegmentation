__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel


res = DataManipulations.collect_vectors('../data/Landmarks/original', '7', 80)

referent = ActiveShapeModel.ReferentModel(res)
referent.align()
referent.rescale_and_realign()

for item in referent.points:
    DataManipulations.Plotter.render_landmarks(item)


