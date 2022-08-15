from typing import Dict, Tuple, Mapping
import numpy
from pandas import DataFrame
from xarray import Dataset, DataArray
from roguewave.interpolate.dataarray import interpolate_track_data_arrray
from roguewave.interpolate.dataframe import interpolate_dataframe_time
from roguewave.interpolate.geometry import Geometry, convert_to_track_set

def interpolate_dataset(
        data_set: Dataset,
        geometry:Geometry,
        periodic_coordinates: Dict[str,float] = None,
        periodic_data: Dict[str, Tuple[float, float]] = None,
        time_variable_in_dataset: str = 'time',
        longitude_variable_in_dataset: str = 'longitude',
        latitude_variable_in_dataset: str = 'latitude',
    ):
    geometry = convert_to_track_set(
        geometry, data_set[time_variable_in_dataset].values)

    if periodic_coordinates is None:
        periodic_coordinates = {longitude_variable_in_dataset: 360}

    for variable in data_set:
        if 'direction' in str(variable).lower():
            periodic_data = {variable:(360,360)}

    out = {}
    for name, track in geometry.tracks.items():
        points = {
            time_variable_in_dataset: track.time,
            longitude_variable_in_dataset: track.longitude,
            latitude_variable_in_dataset:track.latitude
        }
        out[name] = interpolate_at_points(
            data_set=data_set,
            points=points,
            independent_variable=time_variable_in_dataset,
            periodic_coordinates=periodic_coordinates,
            periodic_data=periodic_data
        ).to_dataframe()
        if out[name].index.tz is None:
            out[name].index = out[name].index.tz_localize('utc')
    return out

def interpolate_at_points(
        data_set: Dataset,
        points: Dict[str, numpy.ndarray],
        independent_variable=None,
        periodic_coordinates: Dict[str, float] = None,
        periodic_data: Dict[str, Tuple[float, float]] = None
) -> Dataset:
    if periodic_data is None:
        periodic_data = {}

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
        if variable in periodic_data:
            period_data = periodic_data[variable][0]
            discont = periodic_data[variable][1]
        else:
            period_data = None
            discont = None

        return_data_set[variable] = interpolate_track_data_arrray(
            data_set[variable], points, independent_variable,
            periodic_coordinates=periodic_coordinates,
            period_data=period_data,
            discont=discont
        )
    return return_data_set


def tracks_as_dataset(time, drifter_tracks:Mapping[str,DataFrame],subkey=None):
    tracks = {}
    for track_id, track in drifter_tracks.items():
        if subkey is not None:
            _track = track[subkey]
        else:
            _track = track
        variables = list(_track.columns)

        tracks[track_id] = DataArray(
            interpolate_dataframe_time(_track,time),
            coords={'time': time, 'variables':variables},
            dims=['time','variables'],
            name=track_id,
        )

    dataset = Dataset(tracks).to_array(dim='spotter_id')
    return dataset