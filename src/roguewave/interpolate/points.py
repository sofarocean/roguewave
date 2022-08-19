from typing import Tuple, Callable, List
import numpy
from roguewave.tools.math import wrapped_difference
from roguewave.tools.grid import enclosing_points_1d
from roguewave.interpolate.general import interpolation_weights_1d


def interpolate_points_nd(
        interpolating_coordinates,
        points,
        periodic_coordinates,
        get_data:Callable[ [List[numpy.ndarray]],numpy.ndarray],
        period_data,
        discont,
        output_shape = None,
        full_coordinates = None
        ):
    """

    :param interpolating_coordinates:
    :param points:
    :param periodic_coordinates:
    :param get_data:
    :param period_data:
    :param discont:
    :param output_shape:
    :param full_coordinates:
    :return:
    """


    number_interp_coor = len(interpolating_coordinates)
    number_points = len(points[interpolating_coordinates[0][0]])

    # Find indices and weights for the succesive 1d interpolation problems
    indices_1d = numpy.empty((number_interp_coor, 2, number_points), dtype='int64')
    weights_1d = numpy.empty((number_interp_coor, 2, number_points), dtype='float64')

    for index, (coordinate_name, coordinate) in enumerate(interpolating_coordinates):
        if coordinate_name in periodic_coordinates:
            period = periodic_coordinates[coordinate_name]
        else:
            period = None

        indices_1d[index, :, :] = enclosing_points_1d(
            coordinate, points[coordinate_name], period=period )
        weights_1d[index, :, :] = interpolation_weights_1d(
            coordinate, points[coordinate_name],
            indices_1d[index, :, :], period=period)

    # We keep a running sum of the weights, if a point is excluded because it
    # contains no data (NaN) the weights will no longer add up to 1 - and we
    # reschale to account for the missing value. This is an easy way to account
    # for interpolation near missing points. Note that if the contribution of
    # missing weights ( 1-weights_sum) exceeds 0.5 - we consider the point
    # invalid.



    if output_shape is None:
        output_shape = (number_points,)
    shape =output_shape
    weights_sum = numpy.zeros(shape)
    if period_data is not None:
        interp_val = numpy.zeros(output_shape, dtype=numpy.complex64)
    else:
        interp_val = numpy.zeros(output_shape, dtype=numpy.float64)

    for intp_indices_nd, intp_weight_nd in _next_point(
            number_interp_coor, indices_1d, weights_1d):
        # Loop over all interpolation points one at a time.
        if period_data is not None:
            to_rad = numpy.pi * 2 / period_data
            val = numpy.exp(1j * get_data(intp_indices_nd) * to_rad)
        else:
            try:
                val = get_data(intp_indices_nd)
            except Exception as e:
                raise e

        mask = numpy.isfinite(val)
        ndim = mask.ndim

        while ndim > 1:
            mask = numpy.all( mask,axis=-1 )
            ndim = mask.ndim



        weights_sum[mask,...] += intp_weight_nd[mask]
        interp_val[mask,...] += intp_weight_nd[mask,...] * val[mask,...]

    interp_val = numpy.where(
        weights_sum > 0.5, interp_val / weights_sum, numpy.nan)

    if period_data is not None:
        to_data_units =  period_data / numpy.pi / 2
        interp_val = wrapped_difference(
            delta=numpy.angle(interp_val) * to_data_units,
            period=period_data,
            discont=discont)

    return interp_val


def _next_point(recursion_depth,
                indices_1d: numpy.ndarray,
                weights_1d: numpy.ndarray,
                *narg) \
        -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    We are trying to interpolate over N dimensions. In bilinear interpolation
    this means we have to visit 2**N points. If N is known this is most
    clearly expressed as a set of N nested loops:

    J=-1
    for i1 in range(0,2):
        for i2 in range(0,2):
            ...
                for iN in range(0,2):
                    J+=1
                    do stuff for J'th item.

    Here instead, since we do not know N in advance, use a set of recursive
    loops to depth N, where at the final level we yield for each of the 2**N
    points the values of the points and the weights with which they contribute
    to the interpolated value.

    :param recursion_depth:
    :param indices_1d:
    :param weights_1d:
    :param narg: indices from outer recursive loops
    :return: generater function that yields the J"th thing to do stuff with.
    """
    number_of_coordinates = indices_1d.shape[0]
    number_of_points = indices_1d.shape[2]

    if recursion_depth > 0:
        # Loop over the n'th coordinate, with
        #    n = number_of_coordinates - recursion_depth
        for ii in range(0, 2):
            # Yield from next recursive loop, add the loop coordinate to the
            # arg of the next call
            arg = (*narg, ii)
            yield from _next_point(recursion_depth - 1,
                                   indices_1d, weights_1d, *arg)
    else:
        #
        # Here we construct the "fancy" indexes we will use to grab datavalues.
        indices_nd = []
        weights_nd = numpy.ones((number_of_points,), dtype='float64')
        for index in range(0, number_of_coordinates):
            # get the coordinate index for the current point.
            indices_nd.append( indices_1d[index, narg[index], :] )

            # The N-dimensional weight is the multiplication of all weights
            # of the associated 1d problems
            weights_nd *= weights_1d[index, narg[index], :]

        yield indices_nd, weights_nd