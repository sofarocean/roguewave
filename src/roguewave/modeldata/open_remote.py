"""
    Motivation:
    -----------
        We want to be able to grab multiple variables from the same model
        for the same time period (e.g. a forecast) and return this as a single
        dataset containing all the variables. Complications:
            1. Output is typically stored remotely as one file for one variable
               at one specific valid-time, init-time (sofar).
            2. For a subset of models all variables are stored in a single file
               at one specific valid-time, init-time (ecmf-wam)
            3. For a subset of models multiple valid_times are stored in a
               single file, one variable per file (typical for reanalysis
               products, e.g. era5).

        In all cases we want to abstract away that the files are stored
        remotely through automatic downloading of remote resources, and provide
        a simple caching mechanism that stores files locally, but ensures
        the cache remains of requested size.
"""

from roguewave import filecache
import xarray
from typing import List, Union, Iterable
from datetime import datetime, timedelta, timezone
from .modelinformation import _get_resource_specification
from os import remove, rename
from roguewave.tools import to_datetime
from glob import glob
from .keygeneration import generate_lead_keys, \
    generate_forecast_keys, \
    generate_evaluation_time_keys, \
    generate_analysis_keys
import numpy


# Main functions to interact with module
# =============================================================================
def open_remote_forecast(
        variable: Union[List[str], str],
        init_time: datetime,
        duration: timedelta,
        model_name: str,
        cache_name: str = None) -> xarray.Dataset:
    """
    Get a local dataset associated with a forecast for a given
    model.

    :param variable: name of the variable of interest
    :param init_time: init time of the forecast of interest
    :param duration: maximum lead time of interest
    :param model_name: model name
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return: xarray dataset
    """

    print(f"Get from data from {model_name} for a forecast")
    init_time = init_time.astimezone(tz=timezone.utc)
    return _open_variables(
        variable,
        lambda x: generate_forecast_keys(x, init_time, duration, model_name),
        model_name=model_name,
        cache_name=cache_name
    )


def open_remote_analysis(
        variable: Union[List[str], str],
        start_time: datetime,
        end_time: datetime,
        model_name: str,
        cache_name: str = None) -> xarray.Dataset:
    """
    Get a local dataset associated with the analysis results for
    a given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param model_name: model name
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return: xarray dataset
    """

    print(f"Get data from model {model_name} at anlysis time")
    start_time = _to_utc(start_time)
    end_time = _to_utc(end_time)
    return _open_variables(
        variable,
        lambda x: generate_analysis_keys(
            variable=x,
            start_time=start_time,
            end_time=end_time,
            model_name=model_name
        ),
        model_name=model_name,
        cache_name=cache_name
    )


def open_remote_lead(
        variable: Union[List[str], str],
        start_time: datetime,
        end_time: datetime,
        lead_time: timedelta,
        model_name: str,
        exact: bool = False,
        cache_name: str = None) -> xarray.Dataset:
    """
    Get a local dataset associated with a given lead time for a
    given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param lead_time: Lead time of interest
    :param model_name: model name
    :param exact: If true, generate keys exactly at given lead time. Otherwise
    generate keys as close as possible.
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return: xarray dataset
    """

    print(f"Get data from model {model_name} "
          f"at lead={int(lead_time.seconds/3600)} hour(s)")
    start_time = _to_utc(start_time)
    end_time = _to_utc(end_time)
    return _open_variables(
        variable,
        lambda x: generate_lead_keys(
            variable=x,
            start_time=start_time,
            end_time=end_time,
            lead_time=lead_time,
            model_name=model_name,
            exact=exact
        ),
        model_name=model_name,
        cache_name=cache_name
    )


def open_remote_evaluation(
        variable: Union[List[str], str],
        evaluation_time: datetime,
        model_name: str,
        maximum_lead_time: timedelta = None,
        cache_name: str = None) -> xarray.Dataset:
    """
    Get a local dataset associated with a given evaluation time.

    :param variable: name of the variable of interest
    :param evaluation_time: evaluation time of interest
    :param model_name: model name
    :param maximum_lead_time: maximum lead time of interest
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return: xarray dataset
    """

    print(f"Get data from model {model_name} "
          f"at valid time {evaluation_time.strftime('%Y-%m-%dT%H:%M:%D')}Z")

    evaluation_time = _to_utc(evaluation_time)
    return _open_variables(
        variable,
        lambda x: generate_evaluation_time_keys(
            variable=x,
            evaluation_time=evaluation_time,
            model_name=model_name,
            maximum_lead_time=maximum_lead_time
        ),
        model_name=model_name,
        cache_name=cache_name
    )


# Functions internal to the module.
# =============================================================================


def _open_aws_keys_as_dataset(
        aws_keys: List[str],
        model_variables: List[str],
        single_variables_per_file:bool,
        filetype='netcdf',
        cache_name: str = None,
        concatenation_dimension='time',
        ) -> xarray.Dataset:
    """
    Open a set of remote resources as a single local dataset using a local
    Cache. Datasets will be concatenated along the given concatenation
    dimension.

    :param aws_keys: List of remote resource identifiers ("aws keys")
    :param filetype: filetype of the remote resource
    :param cache_name: name of local file cache
    :param concatenation_dimension: dimension to concatenate datasets along to
        a single dataset.
    :return: dataset concatenated along time dimension
    """

    # Ask for both the files and whether the file was already in cache. We use
    # the latter to do a conversion from grib to netcdf for a freshly
    # downloaded file so we can use a unified netcdf interface and because
    # Grib == Slow.
    def post_process(filepath:str):
        if filetype=='grib':
            _convert_grib_to_netcdf(filepath, model_variables)

    def validate(filepath:str):
        if not single_variables_per_file:
            ds = xarray.open_dataset(filepath,engine='netcdf4')
            for variable in model_variables:
                if variable not in ds:
                    ds.close()
                    return False
            ds.close()
        return True

    # TODO: what happens if we ask the cache to get a something else outside
    #  of this path? will it start to apply the given pre-process functions?
    filecache.set_post_process_function(post_process)
    filecache.set_validate_function(validate)
    filepaths = filecache.filepaths(aws_keys, cache_name )

    datasets = [
        xarray.open_dataset(file,engine='netcdf4', decode_times=False)
            for file in filepaths ]

    # Convert time etc to cf conventions
    datasets = [xarray.decode_cf(dataset) for dataset in datasets ]
    for dataset in datasets:
        init_time = to_datetime(dataset.attrs.get('init_time'))
        if init_time is not None:
            dataset['init_time'] = [numpy.datetime64(init_time)]

    # Concatenate and return resulting dataset
    return xarray.concat(datasets, dim=concatenation_dimension)


def _open_variables(variables,
                    key_generation_function,
                    model_name,
                    cache_name,
                    map_to_sofar_variable_names=True
                    ) -> xarray.Dataset:
    """
    Open the datasets corresponding to the given variables and return as a
    single variable.

    :param variables: variable names in model or Sofar naming convention
    :param key_generation_function: function that takes a variable name as
        input and generates the keys associated with the request.
    :param model_name: name of the model
    :param cache_name: name of the cache we use.
    :param map_to_sofar_variable_names: If true, variable names in the returned
        xarray dataset correspond to Sofar conventions.
    :return: Dataset
    """

    if not isinstance(variables, Iterable) or isinstance(variables,str):
        variables = [variables]

    # Get the model description.
    aws_layout = _get_resource_specification(model_name)

    # We can specify variable names by Sofar name or equivalent model specific
    # name. Here we remap the variable names to make sure they are all
    # specified as model specific names for querying.
    if aws_layout.mapping_variable_name_model_to_sofar is not None:
        model_variable_names = [aws_layout.to_model_variable_name(x)
                                for x in variables]
    else:
        # If no mapping is available, we assume variables already correspond
        # to model names (specifically Sofar variable name conventions).
        model_variable_names = variables

    concatenation_dimension = aws_layout.to_model_variable_name('time')
    if not aws_layout.single_variable_per_file or len(model_variable_names) == 1:
        # For a single variable there is no need to merge datasets. Further,
        # if all variables per valid_time are stored in a the same file
        # (e.g. ECMWF Grib) then we only need to load data once. To note,
        # in that case the key_generation_function will not use the variable
        # name.
        aws_keys = key_generation_function(model_variable_names[0])
        dataset = _open_aws_keys_as_dataset(
            aws_keys=aws_keys,
            filetype=aws_layout.filetype,
            single_variables_per_file=aws_layout.single_variable_per_file,
            cache_name=cache_name,
            concatenation_dimension=concatenation_dimension,
            model_variables=model_variable_names
        )
    else:
        datasets = []
        # If multiple variables are present, and the variables are remotely
        # stored in unique keys, we construct individual datasets for each
        # variable and merge the dataset at the end.
        for variable in model_variable_names:
            aws_keys = key_generation_function(variable)
            datasets.append( _open_aws_keys_as_dataset(
                aws_keys=aws_keys,
                filetype=aws_layout.filetype,
                single_variables_per_file=aws_layout.single_variable_per_file,
                cache_name=cache_name,
                concatenation_dimension=concatenation_dimension,
                model_variables=model_variable_names
                )
            )
        dataset = xarray.merge(datasets)

    # Remap to Sofar variable naming conventions if requested (and needed).
    if aws_layout.mapping_variable_name_model_to_sofar is not None and \
            map_to_sofar_variable_names:
        # we may not have downloaded all variables, make sure we only rename
        # those available in the dataset.
        _map = {key: value for key, value in
                aws_layout.mapping_variable_name_model_to_sofar.items()
                if key in dataset
                }
        dataset = dataset.rename(_map)

    # return the dataset
    return dataset


# Helper Functions
# =============================================================================


def _to_utc(time: datetime) -> datetime:
    """
    Ensure _utc timezone
    :param time: datetime
    :return: datetime with timezone utc.
    """
    return time.astimezone(tz=timezone.utc)


def _convert_grib_to_netcdf(filepath: str, model_variables) -> None:
    """
    Convert a grib file to a netcdf file

    :param filepath: filepath of the grib file
    :return: None
    """
    # open the dataset
    dataset = xarray.open_dataset(filepath, engine="cfgrib",decode_times=False)
    dataset = dataset[model_variables]


    # convert the dataset and close
    dataset.to_netcdf(filepath + '.nc')
    dataset.close()

    # delete old file and any lingering idx files
    remove(filepath)
    if grib_idx_file := glob(filepath + '*' + '.idx'):
        remove(grib_idx_file[0])

    # rename to old file name to ensure consistency.
    rename(filepath + '.nc', filepath)
    return None
