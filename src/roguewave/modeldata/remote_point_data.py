import s3fs
import numpy
from multiprocessing.pool import Pool
from itertools import repeat
import tqdm
from pandas import DataFrame
from roguewave.tools.time import to_datetime64, to_datetime_utc
from roguewave.interpolate.cluster import interpolate_cluster
from xarray import DataArray, open_dataset
from roguewave.interpolate.geometry import ClusterStack, Cluster, Geometry, \
    convert_to_track_set, convert_to_cluster_stack
from typing import Dict,List, Union
from roguewave.modeldata.modelinformation import _get_resource_specification
from roguewave.modeldata.keygeneration import generate_uris
BLOCKSIZE = 1024

def extract_clusters(geometry: Geometry,
                     variable,
                     time_slice,
                     model_name,
                     parallell=False,
                     number_of_workers=10) -> Dict[str, DataFrame]:


    resource = _get_resource_specification(model_name)
    time_base = time_slice.time_base(resource.model_time_configuration)
    valid_time = to_datetime64([ x[0]+x[1] for x in time_base ])

    clusters = convert_to_cluster_stack(geometry, valid_time)

    if isinstance(variable,str):
        variable = [variable]

    out = {  }
    for var in variable:
        uris = generate_uris(var, time_slice, model_name)
        data = _extract_clusters(clusters,uris,var,parallell,
                                 number_of_workers)
        for name, dataframe in data.items():
            if name not in out:
                out[name] = dataframe
            else:
                out[name][var] = dataframe[var]
    return out


def _extract_clusters(clusters: ClusterStack,
                     uris, variable,
                     parallell=False,
                     number_of_workers=10) -> Dict[str, DataFrame]:
    filesystem = s3fs.S3FileSystem()

    # Some properties are assumed homogeneous acrros the files. We can avoid
    # rereading them by caching from the first entry
    with filesystem.open(uris[0], 'rb', cache_type='bytes',
                         block_size=BLOCKSIZE) as file:

        with open_dataset(file, engine='h5netcdf') as dataset:
            latitude = dataset['latitude'].values
            longitude = dataset['longitude'].values

    # Create iterable that serves as input to the worker
    work = zip(clusters.clusters,
               repeat(latitude),
               repeat(longitude),
               repeat(variable),
               uris)

    if parallell:
        with Pool(processes=number_of_workers) as pool:
            data = list(
                tqdm.tqdm(pool.imap(worker, work),
                total=len(clusters))
            )
    else:
        data = list(tqdm.tqdm(map(worker, work)))

    # Transform to a dictionary, with point names as keys and all the relevant
    # values as items
    out = {}
    for index, cluster_data in enumerate(data):
        for point_name, point_value in cluster_data.items():
            if point_name not in out:
                out[point_name] = {'time': [], variable: []}

            out[point_name]['time'].append(
                to_datetime_utc(clusters.time[index]))
            out[point_name][variable].append(point_value)

    _out = {}
    for point_name, point_data in out.items():
        _out[point_name] = DataFrame(index=point_data['time'])
        _out[point_name][variable] = numpy.array(point_data[variable])
    return _out


def worker(arg) -> Dict[str, float]:
    cluster = arg[0]  # type:Cluster
    latitudes = arg[1]
    longitudes = arg[2]
    variable = arg[3]
    uri = arg[4]
    filesystem = s3fs.S3FileSystem()
    out = {}

    for name, point in cluster.points.items():
        if (point.is_valid):
            break
    else:
        # No valid points
        out = {}
        for key in cluster.points:
            out[key] = numpy.nan
        return out

    with filesystem.open(uri, 'rb', cache_type='bytes',
                         blocksize=BLOCKSIZE) as file:

        with open_dataset(file, engine='h5netcdf') as ds:
            scale_factor = get_attr_netcdf(ds.variables[variable].attrs,
                                           'scale_factor', 1)
            FillValue = get_attr_netcdf(ds.variables[variable].attrs,
                                        '_FillValue', numpy.nan)
            missing_value = get_attr_netcdf(ds.variables[variable].attrs,
                                            '_missing_value', numpy.nan)
            add_offset = get_attr_netcdf(ds.variables[variable].attrs,
                                         'add_offset', 0)

            def get_data(indices, _dummy):
                value = ds[variable][
                    ..., DataArray(indices[0]), DataArray(indices[1])].values
                value = numpy.squeeze(value)
                mask = (value == FillValue) | (value == missing_value)
                value[mask] = numpy.nan
                return value * scale_factor + add_offset

            out = interpolate_cluster(
                latitudes,
                longitudes,
                cluster,
                get_data
            )
    return out


def get_attr_netcdf(attrs, name, default):
    # Not all netcdf variable encode all the attributes we are looking for.
    # Here we set default values in case they are missing
    try:
        return attrs[name]
    except:
        return default