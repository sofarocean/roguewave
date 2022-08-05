import numpy
from datetime import datetime
from typing import Dict, Tuple, Hashable, Union, List, Mapping
from xarray import Dataset, DataArray
from pandas import DataFrame
from roguewave.wavespectra import WaveSpectrum1D
from roguewave.metoceandata import MetoceanData


def interpolate_along_spotter_tracks(
        dataset:Dataset,
        spotter_data:Dict[str, Union[Dict,DataFrame,List]],
        time_variable_in_dataset:str = 'time',
        longitude_variable_in_dataset: str = 'longitude',
        latitude_variable_in_dataset: str = 'latitude',
        cyclic_coordinates: dict[str,float] = None,
        default_track_to_use:str = 'wave',
        interpolate_onto_dataset_time:bool = True,
        cyclic_data:Dict[str,Tuple[float,float,float]] = None
        ) -> Dict[str,DataFrame]:
    """
    Function to interpolate data along model
    :param dataset:
    :param spotter_data:
    :param time_variable_in_dataset:
    :param cyclic_coordinates:
    :param default_track_to_use:
    :return:
    """


    output = {}
    for spotter_id, item in spotter_data.items():
        if isinstance(item,Mapping):
            if default_track_to_use in spotter_id:
                data = item[default_track_to_use]
            else:
                raise ValueError(
                    f'No {default_track_to_use} data in the object,'
                    f' please specify from which sensor to use '
                    f' the latitude, longitude and timestamps')
        else:
            data = item

        output[spotter_id] = _interpolate_along_spotter_track(
            dataset=dataset,
            spotter_data=data,
            time_variable_in_dataset=time_variable_in_dataset,
            longitude_variable_in_dataset=longitude_variable_in_dataset,
            latitude_variable_in_dataset=latitude_variable_in_dataset,
            cyclic_coordinates=cyclic_coordinates,
            interpolate_onto_dataset_time=interpolate_onto_dataset_time,
            cyclic_data=cyclic_data
        )

    return output


def _interpolate_along_spotter_track(
        dataset:Dataset,
        spotter_data:Union[DataFrame,List[WaveSpectrum1D],List[MetoceanData]],
        time_variable_in_dataset:str = 'time',
        longitude_variable_in_dataset:str = 'longitude',
        latitude_variable_in_dataset:str = 'latitude',
        cyclic_coordinates: dict[str,float] = None,
        interpolate_onto_dataset_time:bool = True,
        cyclic_data:Dict[str,Tuple[float,float,float]] = None
        ) -> DataFrame:

        if cyclic_coordinates is None:
            cyclic_coordinates = {'longitude':360}

        model_time = dataset['time'].values
        if isinstance(spotter_data, DataFrame):
            track_time = spotter_data.index.values
            track_latitude = spotter_data['latitude'].values
            track_longitude = spotter_data['longitude'].values

        else:
            track_time = numpy.array([numpy.datetime64(x.timestamp)
                                 for x in spotter_data])
            track_latitude = numpy.array([x.timestamp for x in spotter_data]),
            track_longitude = numpy.array([x.timestamp for x in spotter_data])

        # ensure times are in numpy's datetime64, this make standard numerical
        # options possible (interpolate etc.).
        if not numpy.issubdtype(track_time.dtype, numpy.datetime64):
            track_time = numpy.apply_along_axis(lambda x: numpy.datetime64(x),
                                                0, track_time)

        if not numpy.issubdtype(model_time.dtype, numpy.datetime64):
            model_time = numpy.apply_along_axis(lambda x: numpy.datetime64(x),
                                                0, model_time)

        if interpolate_onto_dataset_time:
            track_time = model_time
            # interpolate spotter latitude/longitude tracks to model timebase.
            # For model valid_times where no spotter data is available, values
            # are set to NaN.
            track_latitude = interpolate_track_coordinate_in_time(
                    track_time, track_latitude,model_time)
            track_longitude = interpolate_track_coordinate_in_time(
                    track_time, track_longitude,model_time)

            # Ensure we are not interpolating at model times where there is no
            # observational data
            mask = ~(numpy.isnan(track_latitude) |
                     numpy.isnan(track_longitude))
        else:
            # in this case we might request data at points we have no model
            # data to return- clip to time range for which we do have data.
            mask = (track_time >= model_time[0]) & \
                   (track_time <= model_time[-1])

        track_time = track_time[mask]
        track_latitude = track_latitude[mask]
        track_longitude = track_longitude[mask]

        points = {
            time_variable_in_dataset:track_time,
            latitude_variable_in_dataset: track_latitude,
            longitude_variable_in_dataset: track_longitude
        }

        data = interpolate_points_data_set(
            data_set=dataset,
            points=points,
            independent_variable=time_variable_in_dataset,
            cyclic_coordinates=cyclic_coordinates,
            cyclic_data=cyclic_data
        )
        return data.to_dataframe()

def interpolate_points_data_set(
        data_set: Dataset,
        points: Dict[str, numpy.ndarray],
        independent_variable=None,
        cyclic_coordinates: Dict[str, float] = None,
        cyclic_data:Dict[str,Tuple[float,float,float]] = None
        ):

    if cyclic_data is None:
        cyclic_data = {}

    dimensions = data_set.dims
    if independent_variable is None:
        if 'time' in dimensions:
            independent_variable = 'time'
        else:
            independent_variable = dimensions[0]

    return_data_set = Dataset(
        coords={independent_variable: points[independent_variable]}
    )
    for variable in data_set:
        if variable in cyclic_data:
            cyclic_length_data = cyclic_data[variable][0]
            fractional_cyclic_range = cyclic_data[variable][1:]
        else:
            cyclic_length_data = None
            fractional_cyclic_range = (0,1)

        return_data_set[variable] = interpolate_points_data_arrray(
            data_set[variable], points, independent_variable,
            cyclic_coordinates=cyclic_coordinates,
            cyclic_length_data=cyclic_length_data,
            fractional_cyclic_range=fractional_cyclic_range
        )
    return return_data_set


def interpolate_points_data_arrray(data_array: DataArray,
                                   points: Dict[str, numpy.ndarray],
                                   independent_variable=None,
                                   cyclic_coordinates: Dict[str, float] = None,
                                   cyclic_length_data=None,
                                   fractional_cyclic_range=(0, 1)
                                   ) \
        -> DataArray:
    """
    Interpolate a data array from a dataset at given points.
    :param data_array:
    :param _points:
    :param cyclic_coordinates:
    :return:
    """

    # Get dimensions of the data array
    dimensions = data_array.dims  # type: Tuple[Hashable,...]

    if cyclic_coordinates is not None:
        for wrapped_coordinate in cyclic_coordinates:
            if wrapped_coordinate not in dimensions:
                raise ValueError(f'Cyclic coordinate {wrapped_coordinate} is'
                                 f' not a valid coordinate of the dataset.')
    else:
        cyclic_coordinates = {}

    # number of coordinates
    number_of_coor = len(dimensions)

    if independent_variable is None:
        if 'time' in dimensions:
            independent_variable = 'time'
        else:
            independent_variable = dimensions[0]

    # Ensure that each of the coordinate lists indicated in points is at least
    # 1D
    number_of_points = 0
    for coordinate_name in points:
        points[coordinate_name] = numpy.atleast_1d(points[coordinate_name])
        number_of_points = len(points[coordinate_name])

    if len(points) != len(dimensions):
        dim_str = [str(dim) for dim in dimensions]
        raise ValueError(f' Points expected to have {len(dimensions)} '
                         f'coordinates corresponding to: {", ".join(dim_str)}')

    for coordinate_name in points:
        dim_str = [str(dim) for dim in dimensions]
        if coordinate_name not in dim_str:
            raise ValueError(f' Coordinate {coordinate_name} not in data array'
                             f'coordinates: {", ".join(dim_str)}')

    # Find indices and weights for the succesive 1d interpolation problems
    indices = \
        numpy.empty((number_of_coor, number_of_points, 2), dtype='int64')
    weights = \
        numpy.empty((number_of_coor, number_of_points, 2), dtype='float64')

    for index, coordinate_name in enumerate(data_array.dims):
        # Get weights and indices
        coordinate = data_array[coordinate_name].values
        # If longitude,
        if coordinate_name in cyclic_coordinates:
            cyclic_length = cyclic_coordinates[coordinate_name]
        else:
            cyclic_length = None

        indices[index, :, :], weights[index, :,
                              :] = enclosing_indices_and_weights(
            coordinate, points[coordinate_name], cyclic_length
        )

    output = DataArray(
        numpy.zeros((number_of_points,)),
        coords={independent_variable: points[independent_variable]},
        dims='time'
    )

    output.name = data_array.name
    weights_sum = numpy.zeros((number_of_points,))
    if cyclic_length_data is None:
        total  = numpy.zeros((number_of_points,))
        for value, weight in _recursive_loop_over_coordinates(
                number_of_coor, indices, weights, data_array):
            mask = numpy.isnan(value)
            weight[mask] = 0
            value[mask] = 0
            weights_sum += weight
            total  += value*weight
        output[:] += total / weights_sum
    else:
        # If this is a cyclic variable we are interpolating we will do a vector
        # interpolation of the unit vectors.

        # Calculate the weighted unit vector components.
        cosine = numpy.zeros((number_of_points,))
        sine = numpy.zeros((number_of_points,))
        for value, weight in _recursive_loop_over_coordinates(
                number_of_coor, indices, weights, data_array):
            # ensure the value is in radians.
            mask = numpy.isnan(value)
            weight[mask] = 0
            value[mask] = 0
            weights_sum += weight
            value_radians = value * numpy.pi * 2 / cyclic_length_data
            # add weighted contribution
            cosine += weight * numpy.cos(value_radians)
            sine += weight * numpy.sin(value_radians)
        cosine = cosine/weights_sum
        sine = sine/weights_sum

        # Result is not necessarily unit vector, can be zero. In the latter
        # case the arctan2 simply returns 0.0. Here we will return NaN instead
        # to make explicit it is undefined.
        radius = cosine ** 2 + sine ** 2
        angle = \
            numpy.where(radius > 0, numpy.arctan2(sine, cosine), numpy.nan) \
                * (cyclic_length_data / numpy.pi * 2)

        # Reproject the angle to the desired output domain.
        angle = \
            (angle + cyclic_length_data * fractional_cyclic_range[0]) \
            % cyclic_length_data \
            - cyclic_length_data * fractional_cyclic_range[0]

        output[:] = angle

    return output


def enclosing_indices_and_weights(coordinate, values, cyclic_length=None):
    values = numpy.atleast_1d(numpy.array(values))
    is_cyclic = cyclic_length is not None
    n = len(coordinate)

    if is_cyclic:
        if numpy.abs(coordinate[0] + cyclic_length / 2.) < cyclic_length / 1e6:
            values = (values + cyclic_length / 2
                      ) % cyclic_length - cyclic_length / 2
        else:
            values = values % cyclic_length

        values = numpy.where(values > coordinate[-1],
                             values - cyclic_length, values)
    elif isinstance(values[0], datetime):
        values = numpy.array([numpy.datetime64(x) for x in values])

    indices = numpy.empty((len(values), 2), dtype='int64')
    weights = numpy.empty((len(values), 2), dtype='float64')
    for value_index, value in enumerate(values):
        index = numpy.searchsorted(coordinate, value, side='right')

        if is_cyclic:
            ind1 = (index - 1) % n
            ind2 = index % n
            indices[value_index, :] = (ind1, ind2)

            mesh_delta = (coordinate[ind2] - coordinate[ind1]
                          + cyclic_length / 2) % cyclic_length - cyclic_length / 2

            delta = (value - coordinate[ind1]
                     + cyclic_length / 2) % cyclic_length - cyclic_length / 2
        else:
            ind1 = index - 1
            ind2 = index
            if index == 0:
                mesh_delta = 1
                delta = 1
                indices[value_index, :] = [0, 0]
            elif index == n:
                indices[value_index, :] = [n - 1, n - 1]
                mesh_delta = 1
                delta = 0
            else:
                indices[value_index, :] = [ind1, ind2]
                mesh_delta = (coordinate[ind2] - coordinate[ind1])
                delta = (value - coordinate[ind1])

        frac = delta / mesh_delta
        weights[value_index, :] = (1 - frac, frac)
    return indices, weights


def _recursive_loop_over_coordinates(recursion_depth,
                                     indices: numpy.ndarray,
                                     weights: numpy.ndarray,
                                     dataset: DataArray,
                                     *narg) \
        -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    :param recursion_depth:
    :param indices:
    :param weights:
    :param narg: indices from outer recursive loops
    :return:
    """
    number_of_coordinates = indices.shape[0]
    number_of_points = indices.shape[1]

    if recursion_depth > 0:
        # Loop over the n'th coordinate, with
        #    n = number_of_coordinates - recursion_depth
        for ii in range(0, 2):
            # Execute body
            yield from _recursive_loop_over_coordinates(recursion_depth - 1,
                                                        indices, weights,
                                                        dataset, *narg, ii)
    else:
        #
        return_indices = []
        return_weights = numpy.ones((number_of_points,), dtype='float64')
        for index in range(0, number_of_coordinates):
            return_indices.append(
                DataArray(
                    numpy.squeeze(indices[index, :, narg[index]])
                    # ,dims=[coordinate_names[index]]
                )
            )
            return_weights *= numpy.squeeze(weights[index, :, narg[index]])
        yield dataset[tuple(return_indices)].values, return_weights


def interpolate_track_coordinate_in_time(
        track_time: numpy.ndarray,
        track_lat_or_lon: numpy.ndarray,
        model_time: numpy.ndarray):
    # Interpolate latitude or longitude in time making sure we interpolate
    # correctly across the dateline

    if not numpy.issubdtype( track_time.dtype, numpy.datetime64 ):
        track_time = numpy.apply_along_axis(lambda x:numpy.datetime64(x),
                                            0,track_time)

    if not numpy.issubdtype( model_time.dtype, numpy.datetime64 ):
        model_time = numpy.apply_along_axis(lambda x:numpy.datetime64(x),
                                            0,model_time)

    # Make sure we are in [0,360]
    track_lat_or_lon = track_lat_or_lon % 360

    # unwrap to continuous vector
    track_lat_or_lon = numpy.array(
        numpy.unwrap(track_lat_or_lon, discont=360, period=360))
    model_time = model_time.astype('float64')
    track_time = track_time.astype('float64')

    output = numpy.interp(
        model_time, track_time, track_lat_or_lon, right=numpy.nan,
            left=numpy.nan)

    # interpolate and return to [-180,180) range
    return  (output + 180)%360 - 180
