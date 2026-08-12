import itertools
from typing import Tuple, Callable, List
import numpy
from roguewave.tools.math import wrapped_difference
from roguewave.tools.grid import enclosing_points_1d
from roguewave.interpolate.general import interpolation_weights_1d

_EARTH_RADIUS_KILOMETERS = 6371.0


def _haversine_distance_kilometers(
    latitude_a: numpy.ndarray,
    longitude_a: numpy.ndarray,
    latitude_b: numpy.ndarray,
    longitude_b: numpy.ndarray,
) -> numpy.ndarray:
    latitude_a_radians = numpy.radians(latitude_a)
    latitude_b_radians = numpy.radians(latitude_b)
    delta_latitude_radians = numpy.radians(latitude_b - latitude_a)
    delta_longitude_radians = numpy.radians(longitude_b - longitude_a)

    haversine_term = (
        numpy.sin(delta_latitude_radians / 2) ** 2
        + numpy.cos(latitude_a_radians)
        * numpy.cos(latitude_b_radians)
        * numpy.sin(delta_longitude_radians / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KILOMETERS * numpy.arcsin(numpy.sqrt(haversine_term))


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
        nearest_neighbour=False,
        nan_fallback_radius=0,
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
        self.nearest_neighbour = nearest_neighbour
        self.nan_fallback_radius = nan_fallback_radius

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
                nearest_neighbour=self.nearest_neighbour,
            )

        if self.data_is_periodic:
            return self._periodic_data_interpolator(
                number_points, indices_1d, weights_1d
            )
        else:
            return self._data_interpolator(
                number_points, indices_1d, weights_1d, points
            )

    def _data_interpolator(self, number_points, indices_1d, weights_1d, points):

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
            primary_result = numpy.where(
                weights_sum > 0.5, interp_val / weights_sum, numpy.nan
            )

        if self.nan_fallback_radius == 0 or not numpy.any(weights_sum <= 0.5):
            return primary_result

        failed_point_position, resolved_value = self._radius_neighbor_fallback(
            number_points, indices_1d, weights_1d, weights_sum, points
        )
        if failed_point_position.size == 0:
            return primary_result

        result = primary_result.copy()
        result[self.output_indexing_full(failed_point_position)] = resolved_value
        return result

    def _point_slice(self, output_shaped_array):
        # weights_sum is constant across passive dimensions for a given point,
        # so any single passive index recovers the per-point value.
        indexer = [0] * self.output_ndims
        indexer[self.output_index_coord_index] = slice(None)
        return output_shaped_array[tuple(indexer)]

    def _radius_neighbor_fallback(
        self, number_points, indices_1d, weights_1d, weights_sum, points
    ):
        # For a point whose primary lookup failed, inverse-distance-weight
        # whichever of its radius-N neighbors (in source-grid index space)
        # have valid data instead of returning NaN outright. Everything here
        # is sized to the (typically tiny) failed-point subset, not to
        # number_points -- for a global grid, failed points are a small
        # fraction of the total, and full-size scratch buffers here would
        # scale with the whole grid for no reason.
        #
        # The one point where this assumes the interpolation point axis is
        # output axis 0 (i.e. interp_index_coord_name is the first
        # data_coordinates entry) is the final scatter into result in
        # _data_interpolator, which mirrors the same assumption the primary
        # bilinear loop above already makes.
        if (
            "latitude" not in self.interp_coord_names
            or "longitude" not in self.interp_coord_names
        ):
            raise NotImplementedError(
                "nan_fallback_radius requires 'latitude' and 'longitude' among "
                "the interpolation coordinates"
            )

        coordinate_value_by_name = dict(self.data_coordinates)
        latitude_values = coordinate_value_by_name["latitude"]
        longitude_values = coordinate_value_by_name["longitude"]
        latitude_axis_index = self.interp_coord_names.index("latitude")
        longitude_axis_index = self.interp_coord_names.index("longitude")

        axis_length_by_axis = [
            len(coordinate_value_by_name[coordinate_name])
            for coordinate_name in self.interp_coord_names
        ]
        axis_period_by_axis = [
            self.coordinate_period(coordinate_name)
            for coordinate_name in self.interp_coord_names
        ]

        # Whichever of the two bilinear bracket points carries the larger
        # weight is the nearest source-grid index on that axis.
        bracket_argmax_per_axis = numpy.argmax(weights_1d, axis=1)
        coincident_source_index_per_axis = numpy.take_along_axis(
            indices_1d, bracket_argmax_per_axis[:, None, :], axis=1
        )[:, 0, :]

        # A point outside the source grid's domain gets NaN (not merely low)
        # bilinear weights, and its bracket indices are meaningless clipped
        # edge values -- exclude it here so out-of-domain queries still
        # return NaN instead of a fabricated nearest-edge value.
        point_is_in_domain = numpy.all(numpy.isfinite(weights_1d), axis=(0, 1))
        failed_point_position = numpy.flatnonzero(
            (self._point_slice(weights_sum) <= 0.5) & point_is_in_domain
        )
        if failed_point_position.size == 0:
            return failed_point_position, None

        number_of_failed_points = failed_point_position.size
        passive_shape = tuple(
            int(size)
            for axis, size in enumerate(self.output_shape(number_points))
            if axis != self.output_index_coord_index
        )
        fallback_weight_sum = numpy.zeros(number_of_failed_points)
        fallback_value_sum = numpy.zeros(
            (number_of_failed_points,) + passive_shape, dtype=numpy.float64
        )

        coincident_source_index_of_failed_points = coincident_source_index_per_axis[
            :, failed_point_position
        ]
        # Distances are measured from the actually-requested point, not the
        # coincident source node -- for a general (non-grid-aligned) bilinear
        # miss these are not the same location.
        target_latitude = points["latitude"][failed_point_position]
        target_longitude = points["longitude"][failed_point_position]

        radius = self.nan_fallback_radius
        neighbor_offsets = [
            offset
            for offset in itertools.product(
                range(-radius, radius + 1), repeat=self.interp_ndims
            )
            if any(offset)
        ]

        for neighbor_offset in neighbor_offsets:
            neighbor_source_index_per_axis = (
                coincident_source_index_of_failed_points.copy()
            )
            neighbor_within_bounds = numpy.ones(number_of_failed_points, dtype=bool)
            for axis_index in range(self.interp_ndims):
                neighbor_source_index_per_axis[axis_index] += neighbor_offset[
                    axis_index
                ]
                if axis_period_by_axis[axis_index] is not None:
                    neighbor_source_index_per_axis[axis_index] %= axis_length_by_axis[
                        axis_index
                    ]
                else:
                    neighbor_within_bounds &= (
                        neighbor_source_index_per_axis[axis_index] >= 0
                    ) & (
                        neighbor_source_index_per_axis[axis_index]
                        < axis_length_by_axis[axis_index]
                    )

            if not numpy.any(neighbor_within_bounds):
                continue

            in_bounds_local_position = numpy.flatnonzero(neighbor_within_bounds)
            # Subset to the in-bounds candidates now, once, so every array
            # below (neighbor_value, neighbor_value_is_valid, and the
            # per-axis index lookups) is sized consistently.
            neighbor_source_index_per_axis_in_bounds = neighbor_source_index_per_axis[
                :, neighbor_within_bounds
            ]
            neighbor_query_indices = [
                neighbor_source_index_per_axis_in_bounds[axis_index]
                for axis_index in range(self.interp_ndims)
            ]
            neighbor_value = self.get_data(
                neighbor_query_indices, self.interp_coord_dim_indices
            )
            neighbor_value_is_valid = numpy.all(
                ~numpy.isnan(neighbor_value),
                axis=self.output_passive_coord_dim_indices,
            )
            if not numpy.any(neighbor_value_is_valid):
                continue

            usable_local_position = in_bounds_local_position[neighbor_value_is_valid]

            neighbor_latitude_value = latitude_values[
                neighbor_source_index_per_axis_in_bounds[latitude_axis_index][
                    neighbor_value_is_valid
                ]
            ]
            neighbor_longitude_value = longitude_values[
                neighbor_source_index_per_axis_in_bounds[longitude_axis_index][
                    neighbor_value_is_valid
                ]
            ]
            neighbor_distance_kilometers = _haversine_distance_kilometers(
                target_latitude[usable_local_position],
                target_longitude[usable_local_position],
                neighbor_latitude_value,
                neighbor_longitude_value,
            )
            neighbor_inverse_distance_weight = numpy.where(
                neighbor_distance_kilometers > 0,
                1.0 / neighbor_distance_kilometers,
                0.0,
            )

            usable_value = neighbor_value[neighbor_value_is_valid]
            weight_broadcast_shape = (-1,) + (1,) * (usable_value.ndim - 1)

            fallback_weight_sum[
                usable_local_position
            ] += neighbor_inverse_distance_weight
            fallback_value_sum[usable_local_position] += (
                neighbor_inverse_distance_weight.reshape(weight_broadcast_shape)
                * usable_value
            )

        weight_broadcast_shape = (-1,) + (1,) * (fallback_value_sum.ndim - 1)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            resolved_value = numpy.where(
                fallback_weight_sum.reshape(weight_broadcast_shape) > 0,
                fallback_value_sum
                / fallback_weight_sum.reshape(weight_broadcast_shape),
                numpy.nan,
            )

        return failed_point_position, resolved_value

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
