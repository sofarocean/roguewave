from roguewave.awsfilecache.filecache import create_aws_file_cache,\
    cached_local_aws_files, cache_exists
import xarray
from typing import List

def open_aws_keys_as_dataset( aws_keys:List[str], filetype='netcdf',cache_name:str=None ):

    if cache_name is None:
        cache_name = '__default__'

    if not cache_exists(cache_name):
        create_aws_file_cache(cache_name)

    files = cached_local_aws_files(aws_keys, cache_name)

    if filetype == 'netcdf':
        engine = 'netcdf4'
    elif filetype == 'grib':
        engine = 'cfgrib'
    else:
        engine = filetype

    datasets = []
    for file in files:
        datasets.append(
            xarray.open_dataset(file,engine=engine)
        )
    return xarray.concat( datasets, dim='time' )