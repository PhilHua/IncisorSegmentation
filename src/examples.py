__author__ = 'Sebastijan'

import DataManipulations
import cv2


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

example_read_landmarks_and_plot_over_original_image()