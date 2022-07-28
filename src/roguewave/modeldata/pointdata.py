import s3fs
import h5netcdf
import numpy
from typing import List, Tuple, Union, Dict, TypedDict
from multiprocessing.pool import Pool
from .grid import interpolation_weights, interp_latitude_longitude
from itertools import repeat
from datetime import datetime
import tqdm
from roguewave.wavespectra.wavespectrum import WaveSpectrum
from roguewave.metoceandata import WaveBulkData
from pandas import DataFrame

BLOCKSIZE = 1024


def get_attr_netcdf(attrs, name, default):
    # Not all netcdf variable encode all the attributes we are looking for.
    # Here we set default values in case they are missing
    try:
        return attrs[name]
    except:
        return default


class Cluster(TypedDict):
    aws_key: str
    valid_time: datetime
    variable: str
    points: Dict[str, Tuple[float, float]]


def worker(arg) -> Dict[str, List[Union[datetime, float]]]:
    cluster = arg[0]  # type:Cluster
    latitudes = arg[1]
    longitudes = arg[2]
    variable = cluster['variable']
    filesystem = s3fs.S3FileSystem()
    out = {}

    for key, observation_coordinates in cluster['points'].items():
        if (not numpy.isnan(observation_coordinates[0])) and (
                not (numpy.isnan(observation_coordinates[1]))):
            break
    else:
        # No valid points
        out = {}
        for key in cluster['points']:
            out[key] = (cluster['valid_time'], numpy.nan)
        return out

    with filesystem.open(cluster['aws_key'], 'rb', cache_type='bytes',
                         blocksize=BLOCKSIZE) as file:
        with h5netcdf.File(file, mode='r') as ds:
            scale_factor = get_attr_netcdf(ds.variables[variable].attrs,
                                           'scale_factor', 1)
            FillValue = get_attr_netcdf(ds.variables[variable].attrs,
                                        '_FillValue', numpy.nan)
            missing_value = get_attr_netcdf(ds.variables[variable].attrs,
                                            '_missing_value', numpy.nan)
            add_offset = get_attr_netcdf(ds.variables[variable].attrs,
                                         'add_offset', 0)

            for key, observation_coordinates in cluster['points'].items():
                sum_weight = 0
                out[key] = [cluster['valid_time'], 0]

                if numpy.isnan(observation_coordinates[0]) or numpy.isnan(
                        observation_coordinates[1]):
                    out[key][1] = numpy.nan
                    continue

                weights_and_indices = interpolation_weights(
                    observation_coordinates[0], observation_coordinates[1],
                    latitudes, longitudes)

                for point in weights_and_indices:
                    ilat, ilon = point['lat_lon_indices']
                    value = ds[variable][..., ilat, ilon]
                    if isinstance(value,numpy.ndarray):
                        value = value[0]
                    # If missing/fill continue
                    if value == FillValue or value == missing_value:
                        continue

                    sum_weight += point['weight']
                    out[key][1] = out[key][1] + point['weight'] * value

                if sum_weight == 0.0:
                    out[key][1] = numpy.nan
                else:
                    out[key][1] = out[key][1] / sum_weight
                out[key][1] = out[key][1] * scale_factor + add_offset
    return out


class Track(TypedDict):
    timestamps: List[datetime]
    latitudes: numpy.ndarray
    longitudes: numpy.ndarray


def extract_at_points(
        aws_keys,
        valid_times,
        points_latlons: List[Tuple[float, float]],
        variable_name_in_netcdf,
        parallell=False,
        number_of_workers=10) -> numpy.ndarray:
    # to do the extraction, we convert to a list of the input type the
    # extraction function likes. This is just reorganizing data.
    clusters = []
    for index in range(0, len(aws_keys)):
        #
        aws_key = aws_keys[index]
        valid_time = valid_times[index]
        points = {}
        for index, point in enumerate(points_latlons):
            points[str(index)] = point

        clusters.append(
            Cluster(aws_key=aws_key, valid_time=valid_time,
                    variable=variable_name_in_netcdf,
                    points=points)
        )

    # Call the extraction function and return result.
    _data = extract_clusters(clusters, parallell=parallell,
                             number_of_workers=number_of_workers)

    number_of_points = len(points_latlons)
    number_of_times = len(valid_times)
    data = numpy.empty((number_of_points, number_of_times))

    for key, item in _data.items():
        index = int(key)
        data[index, :] = item['data'][:]

    return data


def extract_along_spotter_tracks(
        aws_keys:List[str],
        valid_times:List[datetime],
        spotter_tracks: Dict[str, Union[List[Union[WaveSpectrum, WaveBulkData]],DataFrame]],
        variable_name_in_netcdf:str,
        parallell=False,
        number_of_workers=10):
    """"""

    tracks = {}
    for spotter_id, spotter in spotter_tracks.items():
        if isinstance(spotter,DataFrame):
            latitudes = spotter['latitude'].values
            longitudes = spotter['longitude'].values
            timestamps = spotter.index
        else:
            latitudes = numpy.array([x.latitude for x in spotter])
            longitudes = numpy.array([x.longitude for x in spotter])
            timestamps = [x.timestamp for x in spotter]
        tracks[spotter_id] =Track(timestamps=timestamps, latitudes=latitudes,
                            longitudes=longitudes)

    return extract_along_tracks(aws_keys, valid_times, tracks,
                                variable_name_in_netcdf, parallell,
                                number_of_workers)


def extract_along_tracks(
        aws_keys,
        valid_times,
        tracks: Dict[str, Track],
        variable_name_in_netcdf,
        parallell=False,
        number_of_workers=10) -> Dict[str, DataFrame]:
    # The tracks are a dictionary, with as key the name of the track (e.g.
    # a Spotter id) and as contents another dictionary (see Track for description)
    # that lists for each track the times, latitudes and longitudes. Since
    # these are not on the model timebase, we first interpolate onto the timebase
    # For valid_times outside the track latitudes/longitudes we set values
    # to NaN
    for key in tracks:
        # convert the observation latitude/longitude vectors to the model
        # timebase
        tracks[key]['latitudes'] = interp_latitude_longitude(
            tracks[key]['timestamps'], tracks[key]['latitudes'], valid_times)
        tracks[key]['longitudes'] = interp_latitude_longitude(
            tracks[key]['timestamps'], tracks[key]['longitudes'], valid_times)

    # to do the extraction, we need to convert the set of tracks across model
    # results to a set of clusters we can extract per model result. This is
    # for efficiency (extract multiple points from the same model time slice)
    clusters = []
    for index in range(0, len(aws_keys)):
        #
        aws_key = aws_keys[index]
        valid_time = valid_times[index]
        points = {}
        for key, item in tracks.items():
            points[key] = (item['latitudes'][index], item['longitudes'][index])

        clusters.append(
            Cluster(aws_key=aws_key, valid_time=valid_time,
                    variable=variable_name_in_netcdf,
                    points=points)
        )

    # Call the extraction function and return result.
    return extract_clusters(clusters, parallell=parallell,
                            number_of_workers=number_of_workers)


def extract_clusters(clusters: List[Cluster],
                     parallell=False,
                     number_of_workers=10) -> Dict[
    str, DataFrame]:
    filesystem = s3fs.S3FileSystem()

    # Some properties are assumed homogeneous acrros the files. We can avoid
    # rereading them by caching from the first entry
    with filesystem.open(clusters[0]['aws_key'], 'rb',
                         cache_type='bytes', block_size=BLOCKSIZE) as file:
        with h5netcdf.File(file, mode='r') as dataset:
            latitude = dataset['latitude'][:]
            longitude = dataset['longitude'][:]

    # Create iterable that serves as input to the worker
    work = zip(clusters, repeat(latitude), repeat(longitude))
    if parallell:
        with Pool(processes=number_of_workers) as pool:
            data = list(
                tqdm.tqdm(pool.imap(worker, work),
                          total=len(clusters)))
    else:
        data = list(tqdm.tqdm(map(worker, work)))

    # Transform to a dictionary, with point names as keys and all the relevant
    # values as items
    out = {}
    for index, point_data in enumerate(data):
        for key, value in point_data.items():
            key: str
            if key not in out:
                out[key] = {'valid_times': [], 'data': []}

            out[key]['valid_times'].append(value[0])
            out[key]['data'].append(value[1])

    _out = {}
    for key, value in out.items():
        _out[key] = DataFrame(index=value['valid_times'])
        _out[key][clusters[0]['variable']] = numpy.squeeze(numpy.array(value['data']))
    return _out
