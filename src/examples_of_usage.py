__author__ = 'Sebastijan'

import cv2

import DataManipulations
import ActiveShapeModel


def example_reading_landmarks_and_display_shape():
    TmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    DataManipulations.Plotter.render_landmarks(TmpObj)


def example_read_landmarks_and_plot_over_original_image():
    TmpObj = DataManipulations.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
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
