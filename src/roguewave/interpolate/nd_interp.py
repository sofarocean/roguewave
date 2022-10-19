from typing import Tuple, Callable, List
import numpy
from roguewave.tools.math import wrapped_difference
from roguewave.tools.grid import enclosing_points_1d
from roguewave.interpolate.general import interpolation_weights_1d


class NdInterpolator:
    def __init__(
        self,
        get_data: Callable[[List[numpy.ndarray], List[int]], numpy.ndarray],
        data_coordinates,
        data_shape,
        interp_coord_names,
        interp_index_coord_name: str,
        data_periodic_coordinates,
        data_period=None,
        data_discont=None,
    ):

        self.get_data = get_data
        self.coord = [x[0] for x in data_coordinates]
        self.data_shape = data_shape
        self.interp_coord_names = interp_coord_names
        self.interp_index_coord_name = interp_index_coord_name
        self.data_coordinates = data_coordinates
        self.data_periodic_coordinates = data_periodic_coordinates
        self.data_period = data_period
        self.data_discont = data_discont

    @property
    def passive_coordinate_names(self):
        return [name for name in self.coord if name not in self.interp_coord_names]

    @property
    def passive_coord_dim_indices(self) -> List[int]:
        return [self.coord.index(x) for x in self.passive_coordinate_names]

    @property
    def output_passive_coord_dim_indices(self) -> Tuple[int]:
        indices = list(range(self.output_ndims))
        _ = indices.pop(indices.index(self.output_index_coord_index))
        return tuple(indices)

    @property
    def interp_coord_dim_indices(self) -> List[int]:
        return [self.coord.index(x) for x in self.interp_coord_names]

    @property
    def interp_index_coord_index(self):
        return self.coord.index(self.interp_index_coord_name)

    def output_shape(self, number_of_points) -> numpy.ndarray:
        output_shape = numpy.ones(self.output_ndims, dtype="int32")
        interpolating_index = self.output_index_coord_index
        passive_ind = self.passive_coord_dim_indices
        jj = 0
        for index in range(self.output_ndims):
            if index == interpolating_index:
                output_shape[index] = number_of_points
            else:
                output_shape[index] = self.data_shape[passive_ind[jj]]
                jj += 1
        return output_shape

    @property
    def output_index_coord_index(self) -> int:
        return numpy.searchsorted(
            self.passive_coord_dim_indices, self.interp_index_coord_index
        )

    @property
    def interpolating_coordinates(self) -> List[Tuple[str, numpy.ndarray]]:
        return [x for x in self.data_coordinates if x[0] in self.interp_coord_names]

    @property
    def output_ndims(self):
        return self.data_ndims - self.interp_ndims + 1

    @property
    def interp_ndims(self):
        return len(self.interp_coord_names)

    @property
    def data_ndims(self):
        return len(self.coord)

    def output_indexing_full(self, slicer):
        indicer = [slice(None)] * self.output_ndims
        indicer[self.output_index_coord_index] = slicer
        return tuple(indicer)

    def output_indexing_broadcast(self, slicer):
        indicer = [None] * self.output_ndims
        indicer[self.output_index_coord_index] = slicer
        return tuple(indicer)

    def coordinate_period(self, coordinate_name):
        if coordinate_name in self.data_periodic_coordinates:
            return self.data_periodic_coordinates[coordinate_name]
        else:
            return None

    @property
    def data_is_periodic(self):
        return self.data_period is not None

    def interpolate(
        self,
        points,
    ):
        """

        :param self:
        :param interpolatinc_loc:
        :param periodic_coordinates:
        :param period_data:
        :param discont:
        :return:
        """
        number_points = len(points[self.interp_coord_names[0]])

        # Find indices and weights for the succesive 1d interpolation problems
        indices_1d = numpy.empty((self.interp_ndims, 2, number_points), dtype="int64")

        weights_1d = numpy.empty((self.interp_ndims, 2, number_points), dtype="float64")

        for index, (coordinate_name, coordinate) in enumerate(
            self.interpolating_coordinates
        ):

            period = self.coordinate_period(coordinate_name)
            indices_1d[index, :, :] = enclosing_points_1d(
                coordinate, points[coordinate_name], period=period
            )
            weights_1d[index, :, :] = interpolation_weights_1d(
                coordinate,
                points[coordinate_name],
                indices_1d[index, :, :],
                period=period,
                extrapolate_left=False,
                extrapolate_right=False,
            )

        if self.data_is_periodic:
            return self._periodic_data_interpolator(
                number_points, indices_1d, weights_1d
            )
        else:
            return self._data_interpolator(number_points, indices_1d, weights_1d)

    def _data_interpolator(self, number_points, indices_1d, weights_1d):

        # We keep a running sum of the weights, if a point is excluded because it
        # contains no data (NaN) the weights will no longer add up to 1 - and we
        # reschale to account for the missing value. This is an easy way to account
        # for interpolation near missing points. Note that if the contribution of
        # missing weights ( 1-weights_sum) exceeds 0.5 - we consider the point
        # invalid.
        output_shape = self.output_shape(number_points)
        weights_sum = numpy.zeros(output_shape)
        interp_val = numpy.zeros(output_shape, dtype=numpy.float64)

        for intp_indices_nd, intp_weight_nd in _next_point(
            self.interp_ndims, indices_1d, weights_1d
        ):
            # Loop over all interpolation points one at a time.
            val = self.get_data(intp_indices_nd, self.interp_coord_dim_indices)

            mask = numpy.all(
                ~numpy.isnan(val), axis=self.output_passive_coord_dim_indices
            ) & (intp_weight_nd > 0)

            weights_sum[self.output_indexing_full(mask)] += intp_weight_nd[
                self.output_indexing_broadcast(mask)
            ]

            interp_val[self.output_indexing_full(mask)] += (
                intp_weight_nd[self.output_indexing_broadcast(mask)]
                * val[self.output_indexing_full(mask)]
            )

        with numpy.errstate(invalid="ignore", divide="ignore"):
            return numpy.where(weights_sum > 0.5, interp_val / weights_sum, numpy.nan)

    def _periodic_data_interpolator(self, number_points, indices_1d, weights_1d):
        # We keep a running sum of the weights, if a point is excluded because it
        # contains no data (NaN) the weights will no longer add up to 1 - and we
        # reschale to account for the missing value. This is an easy way to account
        # for interpolation near missing points. Note that if the contribution of
        # missing weights ( 1-weights_sum) exceeds 0.5 - we consider the point
        # invalid.
        output_shape = self.output_shape(number_points)
        weights_sum = numpy.zeros(output_shape)

        interp_val = numpy.zeros(output_shape, dtype=numpy.complex64)
        for intp_indices_nd, intp_weight_nd in _next_point(
            self.interp_ndims, indices_1d, weights_1d
        ):
            # Loop over all interpolation points one at a time.
            to_rad = numpy.pi * 2 / self.data_period
            val = numpy.exp(
                1j
                * self.get_data(intp_indices_nd, self.interp_coord_dim_indices)
                * to_rad
            )

            mask = numpy.all(
                ~numpy.isnan(val), axis=self.output_passive_coord_dim_indices
            )

            weights_sum[self.output_indexing_full(mask)] += intp_weight_nd[
                self.output_indexing_broadcast(mask)
            ]

            interp_val[self.output_indexing_full(mask)] += (
                intp_weight_nd[self.output_indexing_broadcast(mask)]
                * val[self.output_indexing_full(mask)]
            )

        interp_val = (
            numpy.angle(
                numpy.where(weights_sum > 0.5, interp_val / weights_sum, numpy.nan)
            )
            * self.data_period
            / numpy.pi
            / 2
        )

        return wrapped_difference(
            delta=interp_val, period=self.data_period, discont=self.data_period
        )


def _next_point(
    recursion_depth, indices_1d: numpy.ndarray, weights_1d: numpy.ndarray, *narg
) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
            yield from _next_point(recursion_depth - 1, indices_1d, weights_1d, *arg)
    else:
        #
        # Here we construct the "fancy" indexes we will use to grab datavalues.
        indices_nd = []
        weights_nd = numpy.ones((number_of_points,), dtype="float64")
        for index in range(0, number_of_coordinates):
            # get the coordinate index for the current point.
            indices_nd.append(indices_1d[index, narg[index], :])

            # The N-dimensional weight is the multiplication of all weights
            # of the associated 1d problems
            weights_nd *= weights_1d[index, narg[index], :]

        yield indices_nd, weights_nd
