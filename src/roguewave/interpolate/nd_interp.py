from typing import Tuple, Callable, List, Mapping
import numpy
from roguewave.tools.math import wrapped_difference
from roguewave.tools.grid import enclosing_points_1d
from roguewave.interpolate.general import interpolation_weights_1d

class DataAccesser():
    def __init__(self,
                 get_data:Callable[
                     [List[numpy.ndarray],List[int]],numpy.ndarray],
                 data_coordinates,
                 data_shape,
                 interpolating_loc: Mapping[str, numpy.ndarray],
                 interp_index_coord_name: str
                 ):

        self.get_data = get_data
        self.coord = [ x[0] for x in data_coordinates]
        self.data_shape = data_shape
        self.interpolating_loc = interpolating_loc
        self.interp_index_coord_name = interp_index_coord_name
        self.interp_index_coord_index = 0
        self.data_coordinates = data_coordinates

        # Here we try to do three things:
        #
        # 1) Assert the length of the index coordinate (easy) and assert that
        #    all given path coordinates have the length of the parametrizing
        #    coordinate of the path.
        # 2) Find the ordinal number set (zero based) of the dimension of the
        #    the interpolating coordinate in the m-dimensional data.
        # 3) Find the ordinal number set (zero based) of the trailing
        #    (or passive) coordinates in the interpolation.
        #
        # Implementation wise we assuming everything is a passive coordinate
        # at first, and pop coordinates from the list as we find interpolating
        # coordinates to add to the interpolation coordinate set.
        self.length_index_coordinate = 0
        self.interp_coord_dim_indices = []
        self.passive_coord_dim_indices = list(range(self.data_ndims))
        for intp_coord_name,intp_coord in interpolating_loc.items():
            # Check for 1.
            if self.length_index_coordinate> 0:
                assert len(intp_coord) == self.length_index_coordinate
            self.length_index_coordinate = len(intp_coord)

            # Find the ordinal of the interpolating coordinate
            self.interp_coord_dim_indices.append( self.coord.index(intp_coord_name )
            )
            if intp_coord_name == self.interp_index_coord_name:
                self.interp_index_coord_index = \
                    self.interp_coord_dim_indices[-1]

            # Pop the interpolating coordinate from the passive coordinate
            # list.
            trailing_index = self.passive_coord_dim_indices.index(
                self.interp_coord_dim_indices[-1]
            )
            _  = self.passive_coord_dim_indices.pop(trailing_index)

        # here we determine the shape of the output array - and where the
        # index of the interpolating indexing coordinate is.
        self.output_shape = numpy.ones(self.output_ndims,dtype='int32')
        jj = 0
        self.output_passive_dims = numpy.zeros(self.output_ndims-1,dtype='int32')
        found = False
        for index in range(self.output_ndims):
            if (self.interp_index_coord_index <
                self.passive_coord_dim_indices[jj]) and not found:
                found = True
                self.output_shape[index] = self.length_index_coordinate
                self.output_index_coord_index = index
            else:
                self.output_shape[index] = self.data_shape[
                    self.passive_coord_dim_indices[jj]]
                self.output_passive_dims[jj] = index
                jj+=1


    @property
    def interpolating_coordinates(self) -> List[Tuple[str,numpy.ndarray]]:
        out = []
        for coordinate_name, coordinate in self.data_coordinates:
            if coordinate_name in self.interpolating_loc:
                out.append((coordinate_name, coordinate))
        return out

    @property
    def output_ndims(self):
        return self.data_ndims - self.interp_ndims + 1

    @property
    def interp_ndims(self):
        return len(self.interpolating_loc)

    @property
    def data_ndims(self):
        return len(self.coord)

    def output_indexing_full(self, slicer):
        indicer = [slice(None)] * self.output_ndims
        indicer[self.output_index_coord_index] = slicer
        return tuple(indicer)

    def output_indexing_broadcast(self,slicer):
        indicer = [None] * self.output_ndims
        indicer[self.output_index_coord_index] = slicer
        return tuple(indicer)


def interpolate_points_nd(
        data_accessing:DataAccesser,
        points,
        periodic_coordinates,
        period_data,
        discont,
        ):
    """

    :param data_accessing:
    :param points:
    :param periodic_coordinates:
    :param period_data:
    :param discont:
    :return:
    """


    number_interp_coor = data_accessing.interp_ndims
    number_points = data_accessing.length_index_coordinate

    # Find indices and weights for the succesive 1d interpolation problems
    indices_1d = numpy.empty((number_interp_coor, 2, number_points), dtype='int64')
    weights_1d = numpy.empty((number_interp_coor, 2, number_points), dtype='float64')

    for index, (coordinate_name, coordinate) in enumerate(
            data_accessing.interpolating_coordinates):

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
    weights_sum = numpy.zeros(data_accessing.output_shape)

    if period_data is not None:
        interp_val = numpy.zeros(data_accessing.output_shape,
                                 dtype=numpy.complex64)
    else:
        interp_val = numpy.zeros(data_accessing.output_shape,
                                 dtype=numpy.float64)

    for intp_indices_nd, intp_weight_nd in _next_point(
            number_interp_coor, indices_1d, weights_1d):
        # Loop over all interpolation points one at a time.
        if period_data is not None:
            to_rad = numpy.pi * 2 / period_data
            val = numpy.exp(1j * data_accessing.get_data(
                intp_indices_nd, data_accessing.interp_coord_dim_indices)
                            * to_rad)
        else:
            try:
                val = data_accessing.get_data(intp_indices_nd,
                                    data_accessing.interp_coord_dim_indices)
            except Exception as e:
                raise e

        mask = numpy.all( numpy.isfinite(val),
                          axis=tuple(data_accessing.output_passive_dims))

        weights_sum[data_accessing.output_indexing_full(mask)] += \
            intp_weight_nd[data_accessing.output_indexing_broadcast(mask)]

        interp_val[data_accessing.output_indexing_full(mask)] += \
            intp_weight_nd[data_accessing.output_indexing_broadcast(mask)] \
            * val[data_accessing.output_indexing_full(mask)]

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