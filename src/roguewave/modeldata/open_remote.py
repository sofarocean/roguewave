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
from .modelinformation import _get_resource_specification
from os import remove, rename
from roguewave.tools.time import to_datetime_utc, datetime_from_time_and_date_integers
from glob import glob
from .timebase import TimeSlice
from .keygeneration import generate_uris
import numpy
import pygrib


# Main functions to interact with module
# =============================================================================
def open_remote_dataset(
    variable: Union[List[str], str],
    time_slice: TimeSlice,
    model_name: str,
    cache_name: str = None,
) -> xarray.Dataset:
    """
    Get a local dataset associated with a forecast for a given
    model.

    :param variable: name of the variable of interest
    :param time_slice: time slice of interest.
    :param model_name: model name
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return: xarray dataset
    """

    print(f"Get from data from {model_name} for a forecast")

    return _open_variables(
        variable,
        lambda x: generate_uris(x, time_slice, model_name),
        model_name=model_name,
        cache_name=cache_name,
    )


# Functions internal to the module.
# =============================================================================


def _open_uris_as_dataset(
    aws_keys: List[str],
    model_variables: List[str],
    single_variables_per_file: bool,
    filetype="netcdf",
    cache_name: str = None,
    concatenation_dimension="time",
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
    def postprocess(filepath: str):
        _convert_grib_to_netcdf_pygrib(filepath, model_variables)

    def validate(filepath: str) -> bool:
        if not single_variables_per_file:
            ds = xarray.open_dataset(filepath, engine="netcdf4")
            for variable in model_variables:
                if variable not in ds:
                    ds.close()
                    return False
            ds.close()
        return True

    if filetype == "grib":
        # Add processing for grib files
        filecache.set_directive_function("postprocess", "grib", postprocess)
        filecache.set_directive_function("validate", "grib", validate)
        aws_keys = [f"validate=grib;postprocess=grib:{x}" for x in aws_keys]

    # Load data into cache and get filenames
    filepaths = filecache.filepaths(aws_keys, cache_name)

    if filetype == "grib":
        # Remove processing so that the cache is in the same state as before
        filecache.remove_directive_function("postprocess", "grib")
        filecache.remove_directive_function("validate", "grib")

    datasets = [
        xarray.open_dataset(file, engine="netcdf4", decode_times=False)
        for file in filepaths
    ]

    # Convert time etc to cf conventions
    datasets = [xarray.decode_cf(dataset) for dataset in datasets]
    for dataset in datasets:
        init_time = to_datetime_utc(dataset.attrs.get("init_time"))
        if init_time is not None:
            dataset["init_time"] = [numpy.datetime64(init_time)]

    # Concatenate and return resulting dataset
    return xarray.concat(datasets, dim=concatenation_dimension)


def _open_variables(
    variables,
    key_generation_function,
    model_name,
    cache_name,
    map_to_sofar_variable_names=True,
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

    if not isinstance(variables, Iterable) or isinstance(variables, str):
        variables = [variables]

    # Get the model description.
    aws_layout = _get_resource_specification(model_name)

    # We can specify variable names by Sofar name or equivalent model specific
    # name. Here we remap the variable names to make sure they are all
    # specified as model specific names for querying.
    if aws_layout.mapping_variable_name_model_to_sofar is not None:
        model_variable_names = [aws_layout.to_model_variable_name(x) for x in variables]
    else:
        # If no mapping is available, we assume variables already correspond
        # to model names (specifically Sofar variable name conventions).
        model_variable_names = variables

    concatenation_dimension = aws_layout.to_model_variable_name("time")
    if not aws_layout.single_variable_per_file or len(model_variable_names) == 1:
        # For a single variable there is no need to merge datasets. Further,
        # if all variables per valid_time are stored in a the same file
        # (e.g. ECMWF Grib) then we only need to load data once. To note,
        # in that case the key_generation_function will not use the variable
        # name.
        aws_keys = key_generation_function(model_variable_names[0])
        dataset = _open_uris_as_dataset(
            aws_keys=aws_keys,
            filetype=aws_layout.filetype,
            single_variables_per_file=aws_layout.single_variable_per_file,
            cache_name=cache_name,
            concatenation_dimension=concatenation_dimension,
            model_variables=model_variable_names,
        )
    else:
        datasets = []
        # If multiple variables are present, and the variables are remotely
        # stored in unique keys, we construct individual datasets for each
        # variable and merge the dataset at the end.
        for variable in model_variable_names:
            aws_keys = key_generation_function(variable)
            datasets.append(
                _open_uris_as_dataset(
                    aws_keys=aws_keys,
                    filetype=aws_layout.filetype,
                    single_variables_per_file=aws_layout.single_variable_per_file,
                    cache_name=cache_name,
                    concatenation_dimension=concatenation_dimension,
                    model_variables=model_variable_names,
                )
            )
        dataset = xarray.merge(datasets)

    # Only return requested variables. (for Sofar variables this removes the
    # mask - or MAPSTA - variable from the NetCDF. Since that information is
    # already contained as an exception value in the actual variables this
    # information is superfluous.
    dataset = dataset[model_variable_names]

    # Remap to Sofar variable naming conventions if requested (and needed).
    if (
        aws_layout.mapping_variable_name_model_to_sofar is not None
        and map_to_sofar_variable_names
    ):
        # we may not have downloaded all variables, make sure we only rename
        # those available in the dataset.
        _map = {
            key: value
            for key, value in aws_layout.mapping_variable_name_model_to_sofar.items()
            if (key in dataset)
        }
        dataset = dataset.rename(_map)

    # return the dataset
    return dataset


# Helper Functions
# =============================================================================


def _convert_grib_to_netcdf(filepath: str, model_variables) -> None:
    """
    Convert a grib file to a netcdf file

    :param filepath: filepath of the grib file
    :return: None
    """
    # open the dataset
    dataset = xarray.open_dataset(filepath, engine="cfgrib", decode_times=False)
    dataset = dataset[model_variables]

    # convert the dataset and close
    dataset.to_netcdf(filepath + ".nc")
    dataset.close()

    # delete old file and any lingering idx files
    remove(filepath)
    if grib_idx_file := glob(filepath + "*" + ".idx"):
        remove(grib_idx_file[0])

    # rename to old file name to ensure consistency.
    rename(filepath + ".nc", filepath)
    return None


def _convert_grib_to_netcdf_pygrib(filepath: str, model_variables) -> None:
    """
    Convert a grib file to a netcdf file

    :param filepath: filepath of the grib file
    :return: None
    """

    # Open the grib file
    grib = pygrib.open(filepath)
    coords = {}
    data = {}
    for variable in model_variables:
        # lets get the correct messages - using the given short or long name.
        try:
            messages = grib.select(shortName=variable)
        except ValueError:
            messages = grib.select(name=variable)

        assert len(messages) == 1, "grib files with multiple times not supported"
        message = messages[0]

        # Decode time, valid time
        init_time = datetime_from_time_and_date_integers(
            message["dataDate"], message["dataTime"], as_datetime64=True
        )
        valid_time = datetime_from_time_and_date_integers(
            message["validityDate"], message["validityTime"], as_datetime64=True
        )

        latitude, longitude = message.latlons()
        assert numpy.all(numpy.diff(latitude[0, :]) == 0)

        if variable == model_variables[0]:
            coords = {
                "longitude": longitude[0, :],
                "latitude": latitude[:, 0],
                "validtime": numpy.array(valid_time),
                "time": numpy.array(init_time),
            }

        data[variable] = xarray.DataArray(
            name=variable,
            data=message.values.filled(numpy.nan).astype("float32"),
            dims=["latitude", "longitude"],
            coords={"longitude": longitude[0, :], "latitude": latitude[:, 0]},
        )

    dataset = xarray.Dataset(data_vars=data, coords=coords)
    dataset.to_netcdf(filepath + ".nc")
    dataset.close()

    # delete old file and any lingering idx files
    remove(filepath)
    if grib_idx_file := glob(filepath + "*" + ".idx"):
        remove(grib_idx_file[0])

    # rename to old file name to ensure consistency.
    rename(filepath + ".nc", filepath)
    return None
