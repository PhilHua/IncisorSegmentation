__author__ = 'Sebastijan'

import math


def euclidean_distance(point1, point2):
    """
        Method calculates Euclidean distance between two points

        params:
            point1 = np.array(y1, x1, ...)
            point2 = np.array(y2, x2, ...)
    """

    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))