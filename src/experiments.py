__author__ = 'Sebastijan'


import DataManipulations

res = DataManipulations.collect_vectors('../data/Landmarks/original', '4', 80)

for i in range(1, 15):
    tmoObj = DataManipulations.DataCollector("../data/Landmarks/original/landmarks{}-4.txt".format(i))
    DataManipulations.Plotter.render_landmarks(tmoObj)


