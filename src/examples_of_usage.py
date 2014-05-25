__author__ = 'Sebastijan'

import cv2
import numpy

from DataManipulations import DataCollector, Plotter, collect_vectors, collect_vectors_DataCollector
import ActiveShapeModel
import utils
from Preprocess import Preprocessor


def example_reading_landmarks_and_display_shape():
    TmpObj = DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    Plotter.render_landmarks(TmpObj)


def example_read_landmarks_and_plot_over_original_image():
    TmpObj = DataCollector('../data/Landmarks/original/landmarks1-4.txt')
    img = cv2.imread('../data/Radiographs/01.tif')
    Plotter.render_over_image(TmpObj, img)


def example_collect_landmarks_from_multiple_teeth():
    res = collect_vectors('../data/Landmarks/original', '1', 80)
    print res


def example_calculate_mean_image_and_display():
    res = collect_vectors('../data/Landmarks/original', '5', 80)
    referent = ActiveShapeModel.ReferentModel(res)
    data_coll = DataCollector(None)
    res = referent.mean_model()
    data_coll.read_vector(referent.mean_model())

    Plotter.render_landmarks(data_coll)


def example_translate_to_origin():
    tmpObj = DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    print numpy.mean(tmpObj.points, axis=0)
    tmpObj.translate_to_origin()
    print tmpObj.centroid


def example_scaling_to_unit_and_back():
    tmpObj = DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    print tmpObj.points
    print "*" * 50
    tmpObj.scale_to_unit()
    print "centroid distance: ", tmpObj.check_distance()
    tmpObj.rescale()
    print tmpObj.points


def example_rotating_landmarks():
    tmpObj = DataCollector('../data/Landmarks/original/landmarks1-1.txt')
    Plotter.render_landmarks(tmpObj)

    tmpObj.rotate(1)
    Plotter.render_landmarks(tmpObj)


def example_aligning_model():
    res = collect_vectors('../data/Landmarks/original', '1', 80)

    #aligning the model
    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    #retrieving the mean model
    model = referent.retrieve_mean_model()
    Plotter.render_landmarks(model)


def example_align_model_and_visualize_shapes():
    res = collect_vectors('../data/Landmarks/original', '4', 80)

    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    for shape in referent.points:
        Plotter.render_landmarks(shape)


def example_calculate_principal_components():
    res = collect_vectors('../data/Landmarks/original', '4', 80)

    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    variance = ActiveShapeModel.VarianceModel(referent)
    variance.obtain_components()

    print variance.get_components()
    print "Component variance ratio: ", variance.get_variances_explained()
    print "Eigenvalues: ", variance.get_eigenvalues()


def example_examine_principal_components():

    res = collect_vectors('../data/Landmarks/original', '1', 80)

    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    variance = ActiveShapeModel.VarianceModel(referent)
    variance.obtain_components()

    components = variance.get_components()
    eigenvals = variance.get_eigenvalues()

    shapes = utils.vary_component(referent.mean_shape, components.transpose(), eigenvals, 1, 10)

    tmpObj = DataCollector(None)
    Plotter.render_landmarks(referent.mean_shape)

    for ind in range(len(shapes)):
        tmpObj.read_vector(shapes[ind, :])
        Plotter.render_landmarks(tmpObj)


def example_clean_image():
    img = cv2.imread('../data/Radiographs/01.tif')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_top = Preprocessor.top_hat_transform(img)
    img_bottom = Preprocessor.bottom_hat_transform(img)

    Plotter.display_image(img, 'Original image')
    Plotter.display_image(img_top, 'Top hat filtered')
    Plotter.display_image(img_bottom, 'Bottom hat filtered')
    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)

    Plotter.display_image(img, 'Result')


def example_using_fourier():
    img = cv2.imread('../data/Radiographs/01.tif', 0)
    Plotter.display_image(img, 'Original image')

    dft_shift = Preprocessor.calculate_fourier(img)
    dft_shift = Preprocessor.high_pass_filter(dft_shift, img.shape, v_offset=7, h_offset=7)

    dft_shift = Preprocessor.low_pass_filter(dft_shift, img.shape, v_offset=120, h_offset=120)

    img_back = Preprocessor.inverse_fourier_transform(dft_shift)
    Preprocessor.display_fourier(img_back)

    img_edges = Preprocessor.find_edges(img_back)
    Preprocessor.display_fourier(img_edges)


def example_normals():
    res = collect_vectors('../data/Landmarks/original', '5', 80)

    referent = ActiveShapeModel.ReferentModel(res)
    referent.align()
    referent.rescale_and_realign()

    variance = ActiveShapeModel.VarianceModel(referent)
    variance.obtain_components()

    asm = ActiveShapeModel.ActiveShape(cv2.imread('../data/Radiographs/01.tif'), (857, 1359), variance)
    asm._calculate_normals()
    Plotter.render_normals(asm)


def example_sample_around_points():
    def preprocess(image_path):
        img = cv2.imread(image_path, 0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
        return numpy.sqrt(sobelx**2 + sobely**2)

    res1, images = collect_vectors_DataCollector('../data/Landmarks/original', '1', 80)
    images = ['../data/Radiographs/' + x for x in images]

    img = preprocess(images[0])
    sample = ActiveShapeModel.Sampler(img, 3, res1[0])
    out = sample.sample()

    print out
    print out.shape
