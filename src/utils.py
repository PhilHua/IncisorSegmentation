__author__ = 'Sebastijan'

import math
import numpy


def euclidean_distance(point1, point2):
    """
        Method calculates Euclidean distance between two points

        params:
            point1 = np.array(y1, x1, ...)
            point2 = np.array(y2, x2, ...)
    """

    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))


def rotation_alignment(referent_shape, current_shape):
    """
        Method calculates rotation alignment of two shape

        params:
            referent_shape : fixed shape  -- DataCollector type
            current_shape : shape to be aligned -- DataCollector type

        returns:
            angle in radians
    """
    numerator = 0.
    denominator = 0.

    for i in range(len(referent_shape.points)):
        numerator += current_shape.points[i, 0] * referent_shape.points[i, 1] - current_shape.points[i, 1] * referent_shape.points[i, 0]
        denominator += current_shape.points[i, 0] * referent_shape.points[i, 0] + current_shape.points[i, 1] * referent_shape.points[i, 1]

    return math.atan2(numerator, denominator)


def is_converged(old_vector, new_vector, threshold=0.00001):
    """
        Method compares two mean shapes, one from previous iteration and second one from the current iteration
        -- converging threshold is set on the maximum change of component
    """
    diff = new_vector.points - old_vector.points
    diff = numpy.power(diff, 2)
    maximum = numpy.max(diff)

    #print "new vector: ", new_vector.as_vector()
    #print "old vector: ", old_vector.as_vector()
    print "new difference: ", maximum
    print "*" * 30

    #a = raw_input("Dalje?")

    if maximum > threshold:
        return False
    else:
        return True


def cvt_vector_to_points(vector, num, dim):
    """
        Method converts vector of [y1, x1, ..., yn, xn] to point-wise matrix

        params:
            vector:  should be 1xn numpy array
    """

    return numpy.reshape(vector, (num, dim))


def cvt_points_to_vector(points):
    """
        Method converts point-wise matrix to vector of [y1, x1, ..., yn, xn]

        params:
            points : numpy matrixof points
    """

    return numpy.hstack(points)