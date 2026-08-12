import numpy
from roguewave.interpolate.nd_interp import NdInterpolator


def _make_get_data(grid_value, grid_mask):
    def get_data(indices, _dimension_indices):
        latitude_index, longitude_index = indices[0], indices[1]
        value = grid_value[latitude_index, longitude_index].astype(float).copy()
        value[~grid_mask[latitude_index, longitude_index]] = numpy.nan
        return value

    return get_data


def _make_interpolator(
    grid_value, grid_mask, nan_fallback_radius, periodic_longitude=None
):
    latitude_values = numpy.array([-1.0, 0.0, 1.0])
    longitude_values = numpy.array([-1.0, 0.0, 1.0])
    periodic_coordinates = (
        {"longitude": periodic_longitude} if periodic_longitude is not None else {}
    )

    return NdInterpolator(
        get_data=_make_get_data(grid_value, grid_mask),
        data_coordinates=(
            ("latitude", latitude_values),
            ("longitude", longitude_values),
        ),
        data_shape=[3, 3],
        interp_coord_names=["latitude", "longitude"],
        interp_index_coord_name="latitude",
        data_periodic_coordinates=periodic_coordinates,
        nan_fallback_radius=nan_fallback_radius,
    )


# Grid layout (row=latitude index, column=longitude index), matching
# latitude_values=[-1, 0, 1] and longitude_values=[-1, 0, 1]:
#   SW  S  SE       10  20  30
#   W   C  E    =   40 100  50
#   NW  N  NE       60  70  80
GRID_VALUE = numpy.array([[10.0, 20.0, 30.0], [40.0, 100.0, 50.0], [60.0, 70.0, 80.0]])


def test_coincident_point_valid_ignores_fallback():
    grid_mask = numpy.ones((3, 3), dtype=bool)
    interpolator = _make_interpolator(GRID_VALUE, grid_mask, nan_fallback_radius=1)

    result = interpolator.interpolate(
        {"latitude": numpy.array([0.0]), "longitude": numpy.array([0.0])}
    )

    assert result[0] == 100.0


def test_masked_coincident_point_blends_valid_radius_one_neighbors():
    grid_mask = numpy.zeros((3, 3), dtype=bool)
    grid_mask[0, 1] = True  # S, value 20
    grid_mask[2, 1] = True  # N, value 70, equidistant from the target as S

    interpolator = _make_interpolator(GRID_VALUE, grid_mask, nan_fallback_radius=1)

    result = interpolator.interpolate(
        {"latitude": numpy.array([0.0]), "longitude": numpy.array([0.0])}
    )

    assert numpy.abs(result[0] - 45.0) < 1e-6


def test_masked_coincident_point_with_no_valid_neighbors_stays_nan():
    grid_mask = numpy.zeros(
        (3, 3), dtype=bool
    )  # every point masked, including the target

    interpolator = _make_interpolator(GRID_VALUE, grid_mask, nan_fallback_radius=1)

    result = interpolator.interpolate(
        {"latitude": numpy.array([0.0]), "longitude": numpy.array([0.0])}
    )

    assert numpy.isnan(result[0])


def test_default_radius_zero_preserves_current_nan_behavior():
    grid_mask = numpy.zeros((3, 3), dtype=bool)
    grid_mask[0, 1] = True
    grid_mask[2, 1] = True

    interpolator = _make_interpolator(GRID_VALUE, grid_mask, nan_fallback_radius=0)

    result = interpolator.interpolate(
        {"latitude": numpy.array([0.0]), "longitude": numpy.array([0.0])}
    )

    assert numpy.isnan(result[0])


def test_fallback_wraps_around_periodic_longitude():
    latitude_values = numpy.array([0.0])
    longitude_values = numpy.array([0.0, 1.0, 2.0])
    grid_value = numpy.array([[90.0, 100.0, 10.0]])
    grid_mask = numpy.array([[True, False, True]])  # target (index 1) masked

    interpolator = NdInterpolator(
        get_data=_make_get_data(grid_value, grid_mask),
        data_coordinates=(
            ("latitude", latitude_values),
            ("longitude", longitude_values),
        ),
        data_shape=[1, 3],
        interp_coord_names=["latitude", "longitude"],
        interp_index_coord_name="latitude",
        data_periodic_coordinates={"longitude": 3.0},
        nan_fallback_radius=1,
    )

    result = interpolator.interpolate(
        {"latitude": numpy.array([0.0]), "longitude": numpy.array([1.0])}
    )

    # Both radius-1 neighbors (index 0 to the west, index 2 wrapping around
    # to the east) are valid and equidistant, so the result is their average.
    assert numpy.abs(result[0] - 50.0) < 1e-6
