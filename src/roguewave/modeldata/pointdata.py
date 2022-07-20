import s3fs
import h5netcdf
import numpy
from typing import List, Tuple, Union, Dict
from multiprocessing.pool import Pool
from .grid import interpolation_weights
from itertools import repeat
from datetime import datetime
import tqdm
from typing import TypedDict

BLOCKSIZE = 1024


def get_attr_netcdf(attrs, name, default):
    # Not all netcdf variable encode all the attributes we are looking for.
    # Here we set default values in case they are missing
    try:
        return attrs[name]
    except:
        return default


class _Description(TypedDict):
    aws_key: str
    valid_time: datetime
    variable: str
    points: Dict[str, Tuple[float, float]]


def worker(arg)->Dict[str,Tuple[datetime,float]]:
    description = arg[0]  # type:_Description
    latitudes = arg[1]
    longitudes = arg[2]
    variable = description['variable']
    filesystem = s3fs.S3FileSystem()
    out = {}

    with filesystem.open(description['aws_key'], 'rb', cache_type='bytes',
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

            for key, observation_coordinates in description['points'].items():
                sum_weight = 0
                out[key] = (description['valid_time'],0)

                weights_and_indices = interpolation_weights(
                    observation_coordinates[0], observation_coordinates[1],
                    latitudes, longitudes)

                for point in weights_and_indices:
                    ilat, ilon = point['lat_lon_indices']
                    value = ds[variable][..., ilat, ilon]

                    # If missing/fill continue
                    if value == FillValue or value == missing_value:
                        continue

                    sum_weight += point['weight']
                    out[key][1] = out[key][1] + point['weight'] * value

                if sum_weight == 0.0:
                    out[key][1] = numpy.nan
                else:
                    out[key] = out[key] / sum_weight
                out[key][1] = out[key][1] * scale_factor + add_offset
    return out

    # Dict[str,List[tuple[datetime,float,float]]]

def remote_data_at_points(aws_keys,
                          valid_times,
                          observation_locations,
                          parallell=False,
                          number_of_workers=10) -> Dict[str,Dict[str,numpy.ndarray]]:
    return _extract_points()

def _extract_points(points_to_extract: List[_Description],
                          parallell=False,
                          number_of_workers=10) -> Dict[str,Dict[str,numpy.ndarray]]:
    filesystem = s3fs.S3FileSystem()

    # Some properties are assumed homogeneous acrros the files. We can avoid
    # rereading them by caching from the first entry
    with filesystem.open(points_to_extract[0]['aws_key'], 'rb',
                         cache_type='bytes', block_size=BLOCKSIZE) as file:
        with h5netcdf.File(file, mode='r') as dataset:
            latitude = dataset['latitude'][:]
            longitude = dataset['longitude'][:]

    # Create iterable that serves as input to the worker
    work = zip(points_to_extract, repeat(latitude), repeat(longitude))
    if parallell:
        with Pool(processes=number_of_workers) as pool:
            data = list(
                tqdm.tqdm(pool.imap(worker, work),
                          total=len(points_to_extract)))
    else:
        data = list(tqdm.tqdm(map(worker, work)))

    # Transform to a dictionary, with point names as keys and all the relevant
    # values as items
    out = {}
    for index, point_data in enumerate(data):
        for key,value in point_data.items():
            key:str
            if key not in out:
                out[key] = {'valid_times':[],'data': []}

            out[key]['valid_time'].append(value[0])
            out[key]['data'].append(value[1])

    _out = {}
    for key, value in out.items():
        _out[key] = {}
        _out[key]['valid_time'] = numpy.array(value['valid_time'] )
        _out[key]['data'] = numpy.array(value['data'])
    return _out
