__author__ = 'Sebastijan'

import cv2
import numpy
import DataManipulations
import ActiveShapeModel


def example_reading_landmarks_and_display_shape():
    TmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    DataManipulations.Plotter.render_landmarks(TmpObj)


def example_read_landmarks_and_plot_over_original_image():
    TmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-4.txt')
    img = cv2.imread('../data/Radiographs/01.tif')
    DataManipulations.Plotter.render_over_image(TmpObj, img)


def example_collect_landmarks_from_multiple_teeth():
    res = DataManipulations.collect_vectors('../data/Landmarks/original', '1', 80)
    print res


def example_calculate_mean_image_and_display():
    res = DataManipulations.collect_vectors('../data/Landmarks/original', '5', 80)
    referent = ActiveShapeModel.ReferentModel(res)
    data_coll = DataManipulations.DataCollector(None)
    res = referent.mean_model()
    data_coll.read_vector(referent.mean_model())

    DataManipulations.Plotter.render_landmarks(data_coll)


def example_translate_to_origin():
    tmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    print numpy.mean(tmpObj.points, axis=0)
    tmpObj.translate_to_origin()
    print tmpObj.centroid


def example_scaling_to_unit_and_back():
    tmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    print tmpObj.points
    print "*" * 50
    tmpObj.scale_to_unit()
    print "centroid distance: ", tmpObj.check_distance()
    tmpObj.rescale()
    print tmpObj.points


def example_rotating_landmarks():
    tmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    DataManipulations.Plotter.render_landmarks(tmpObj)

    tmpObj.rotate(1)
    DataManipulations.Plotter.render_landmarks(tmpObj)


def example_aligning_model():
    res = DataManipulations.collect_vectors('../data/Landmarks/original', '1', 80)

    #aligning the model
    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    #retrieving the mean model
    model = referent.retrieve_mean_model()
    DataManipulations.Plotter.render_landmarks(model)


def example_align_model_and_visualize_shapes():
    res = DataManipulations.collect_vectors('../data/Landmarks/original', '4', 80)

    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    for shape in referent.points:
        DataManipulations.Plotter.render_landmarks(shape)