from typing import Dict
import numpy
from xarray import DataArray

from roguewave.interpolate.nd_interp import NdInterpolator


def interpolate_track_data_arrray(
    data_array: DataArray,
    tracks: Dict[str, numpy.ndarray],
    independent_variable=None,
    periodic_coordinates: Dict[str, float] = None,
    period_data=None,
    discont=None,
) -> DataArray:
    """
    Interpolate a data array from a dataset at given points in N-dimensions.

    The interpolation function takes care of the following edge cases not
    covered by standard xarray interpolation:

    1) Interpolation along cyclic dimensions (e.g. Longitude interpolating
       across prime- or anti-meridian)
    2) Interpolation of cyclic data. E.g. directions.
    3) Interpolation near non-existing values (e.g. NaN values representing
       land).

    When these are covered by standard xarray interpolation we should
    definitely switch.

    :param data_array: xarray data array.

    :param _points: dictionary with data_array coordinate names as keys and
                    a numpy array with N entries. With coordiantes lat/lon:
                    {
                        lat: [ lat0, lat1 ,lat2, .., 10]
                        lon: [ [lon0 ,lon1,lon2, .., lonN]
                        time: [t0,t1,t2, ... tN]
                    }
                    where the points we are interolating are formed by the
                    tuples [  (t[0],lat[0],lon[0]),
                                 ....
                              (t[1],lat[N],lon[N]),
                    ]

    :param independent_variable: coordinate that is used to parametrize the
        points, usually time. This variable is used as the output coordinate
        for the returned data array. If None and 'time' is a valid coordinate
        'time' is used, otherwise it defaults to whatever the first returned
        coordinate is from data_array.dims

    :param periodic_coordinates: Dictonary that contains as keys the coordinate
        names of cyclic coordinates, and as value the cyclic length of the
        coordinate.

    :param period_data: Cyclic length of the data. If data is not cyclic
        this is set to None (default).

    :param fractional_cyclic_range: range of the cyclic data expressed as a
        fraciton of the cyclic lenght. E.g for angles in [-pi,pi] with
        cyclic length 2*pi we would give [-0.5,0.5]. Defaults to [0,1].

    :return: A data array of interpolated values with as coordinates the
        specified independent coordinate (usually time).
    """

    # Get dimensions of the data array
    dimensions = data_array.dims

    if periodic_coordinates is not None:
        for wrapped_coordinate in periodic_coordinates:
            if wrapped_coordinate not in dimensions:
                raise ValueError(
                    f"Cyclic coordinate {wrapped_coordinate} is"
                    f" not a valid coordinate of the dataset."
                )
    else:
        periodic_coordinates = {}

    if independent_variable is None:
        if "time" in dimensions:
            independent_variable = "time"
        else:
            independent_variable = dimensions[0]

    # Ensure that each of the coordinate lists indicated in points is at least
    # 1D
    for coordinate_name in tracks:
        tracks[coordinate_name] = numpy.atleast_1d(tracks[coordinate_name])

    if len(tracks) != len(dimensions):
        dim_str = [str(dim) for dim in dimensions]
        raise ValueError(
            f" Points expected to have {len(dimensions)} "
            f'coordinates corresponding to: {", ".join(dim_str)}'
        )

    for coordinate_name in tracks:
        dim_str = [str(dim) for dim in dimensions]
        if coordinate_name not in dim_str:
            raise ValueError(
                f" Coordinate {coordinate_name} not in data array"
                f'coordinates: {", ".join(dim_str)}'
            )

    coordinates = []
    for index, coordinate_name in enumerate(data_array.dims):
        # Get weights and indices
        coordinates.append((coordinate_name, data_array[coordinate_name].values))

    def _get_data(indices, _dummy):
        return data_array[tuple([DataArray(x) for x in indices])].values

    interpolator = NdInterpolator(
        get_data=_get_data,
        data_coordinates=coordinates,
        data_shape=data_array.shape,
        interp_coord_names=list(tracks.keys()),
        interp_index_coord_name=independent_variable,
        data_periodic_coordinates=periodic_coordinates,
        data_period=period_data,
        data_discont=discont,
    )
    interpolated_values = interpolator.interpolate(tracks)

    return DataArray(
        interpolated_values,
        coords={independent_variable: tracks[independent_variable]},
        dims=independent_variable,
        name=data_array.name,
    )
