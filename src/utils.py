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


def is_converged(old_vector, new_vector, threshold=0.0000001):
    """
        Method compares two mean shapes, one from previous iteration and second one from the current iteration
        -- converging threshold is set on the maximum change of component
    """
    diff = new_vector.points - old_vector.points
    #diff = numpy.power(diff, 2)
    diff = diff**2
    maximum = numpy.max(diff)

    #print "new vector: ", new_vector.as_vector()
    #print "old vector: ", old_vector.as_vector()
    #print "new difference: ", maximum
    #print "*" * 30

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
            points : numpy matrix of points
    """

    return numpy.hstack(points)


def vary_component(mean_model, components, eigenvalues, com_no, number_of_interpolations):
    """
        Method varies mean_model landmark points in a direction of a principal component chosen with com_no

        @params:
            mean_model : DataCollector object representing the mean model shape
            components : matrix of principal components; numpy array of (n_dimensions, n_components) form
            eigenvals : eigenvalues corresponding to the principal components
            com_no : principal component to vary
            number_of_interpolations : number of shapes to interpolate between -3*sqrt(eigenvalue_com_no) and 3*sqrt(eigenvalue_com_no)
    """

    shapes = numpy.zeros((number_of_interpolations, len(components)))
    step = 2. * 3. * math.sqrt(eigenvalues[com_no])/number_of_interpolations

    for step_ind in range(number_of_interpolations):
        b = numpy.zeros((len(components[0]), 1))
        b[com_no] = -3. * math.sqrt(eigenvalues[com_no]) + step * step_ind

        shapes[step_ind, :] = mean_model.as_vector() + components.dot(b).transpose()

    return shapes


def normal(point_one, point_two):
    """
        The function calculates the normal to a line determined with point_one and point_two
            Ax + By + C = 0
            n = (A, B)

        @params:
            point_one, point_two : (y1, x1), (y2, x2)
    """
    return numpy.array([point_one[1] - point_two[1], point_two[0] - point_one[0]])

