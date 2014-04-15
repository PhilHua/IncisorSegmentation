__author__ = 'Sebastijan'


import DataManipulations
import ActiveShapeModel

tmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks7-4.txt')
DataManipulations.Plotter.render_landmarks(tmpObj)

res = DataManipulations.collect_vectors('../data/Landmarks/original', '4', 80)

referent = ActiveShapeModel.ReferentModel(res)
referent.align()

model = referent.retrieve_mean_model()
model.rescale()
model.realign_to_absolute()
DataManipulations.Plotter.render_landmarks(model)