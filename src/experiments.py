__author__ = 'Sebastijan'


import data_manipulation
import cv2

TmpObj = data_manipulation.DataCollector('../data/Landmarks/original/landmarks1-1.txt')
img = cv2.imread('../data/Radiographs/01.tif')
TmpObj.render_over_image(img)

res = data_manipulation.collect_vectors('../data/Landmarks/original', '1', 80)
print res