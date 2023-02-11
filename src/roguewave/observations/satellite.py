"""
Contents: Routines to get data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to calculate times we have output at different init and forecasthours

Functions:

- `get_satellite_data`, get observational data from available satellites
- `get_satellite_available`, get list of available satellites
"""


# Import
# =============================================================================
from typing import Sequence, Literal, List, Dict, Union
from datetime import datetime, timedelta
from pandas import DataFrame
from roguewave.modeldata.timebase import ModelTimeConfiguration, TimeSliceLead
import boto3
from roguewave import filecache
import xarray
import netCDF4
import pandas
import numpy
from os import remove, rename


SATELLITES = Literal["jason-3", "saral", "sentinal-6"]


satellite_formats = {
    "jason-3": {
        "prefix": "JA3",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windVelocity10Meter",
            "swh_ocean": "significantWaveHeight",
            "time": "time",
            "latitude": "latitude",
            "longitude": "longitude",
        },
    },
    "sentinal-6": {
        "prefix": "S6A",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windVelocity10Meter",
            "swh_ocean": "significantWaveHeight",
            "time": "time",
            "latitude": "latitude",
            "longitude": "longitude",
        },
    },
    "saral": {
        "prefix": "SRL",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windVelocity10Meter",
            "swh": "significantWaveHeight",
            "time": "time",
            "lat": "latitude",
            "lon": "longitude",
        },
    },
}


def _get_from_cache(
    aws_keys, output_sampling_interval: timedelta, inv_mapping, variables
) -> pandas.DataFrame:
    """
    Get data from the cache (if available) or download otherwise.

    :param aws_keys: s3 aws keys
    :param output_sampling_interval: desired sample rate of output (if None, use native sample rate)
    :param inv_mapping: variable name mapping from name in netcdf stored remotely to Sofar name
    :param variables: which variables we want to extract from the remote files (Sofar convention)
    :return:
    """

    def postprocess(filepath: str):
        data = _open_grouped_netcdf_file(filepath, inv_mapping, variables)

        if output_sampling_interval is not None:
            data = _resample_dataframe(data, output_sampling_interval)

        dataset = xarray.Dataset.from_dataframe(data)
        dataset.to_netcdf(filepath + ".nc", engine="netcdf4")
        dataset.close()
        remove(filepath)
        rename(filepath + ".nc", filepath)
        return None

    def validate(filepath: str) -> bool:
        data = xarray.open_dataset(filepath)
        for var in variables:
            if var not in data:
                data.close()
                return False
        data.close()
        return True

    if output_sampling_interval is None:
        sampling = "None"
    else:
        sampling = str(output_sampling_interval.total_seconds())

    # Add processing for grib files
    filecache.set_directive_function("postprocess", f"satellite{sampling}", postprocess)
    filecache.set_directive_function("validate", f"satellite{sampling}", validate)
    aws_keys = [
        f"validate=satellite{sampling};postprocess=satellite{sampling}:{x}"
        for x in aws_keys
    ]

    # Add a "comment" to the uri to make it unique for the given sampling. If we change sampling
    # we need to re-download the original files.
    comment = f"<<{sampling}" + "_".join(variables)
    # print(comment)
    aws_keys = [key + comment for key in aws_keys]

    # Load data into cache and get filenames

    filepaths = filecache.filepaths(aws_keys)

    # Remove processing so that the cache is in the same state as before
    filecache.remove_directive_function("postprocess", f"satellite{sampling}")
    filecache.remove_directive_function("validate", f"satellite{sampling}")

    out = []
    for file in filepaths:
        data = xarray.open_dataset(file)
        # Check if the dataset contains any data- if not skip it. Xarray will otherwise crash in trying to open it.
        if data.dims["index"] < 1:
            data.close()
            continue

        out.append(data.to_dataframe())
        data.close()
        # out.append(xarray.open_dataset( file ).to_dataframe())

    if len(out) < 1:
        return None

    else:
        # Concatenate individual data frames into a single dataframe for the satellite
        return pandas.concat(out)


def get_satellite_data(
    start_date: datetime,
    end_date: datetime,
    satellites: Sequence[SATELLITES] = None,
    variables=("significantWaveHeight", "windVelocity10Meter"),
    output_sampling_interval=None,
) -> DataFrame:
    """
    Download satellite data that is stored on s3 for the given interval specificed by start date and end date.

    :param start_date: start date of interval
    :param end_date: end date of interval
    :param satellites: List of satellites to download data from - options are ['jason-3', 'saral', 'sentinal-6'], if
        None is provided as input (default) data from all available satellites is downloaded.
    :param variables: List of Variables to download. Currently only ('significantWaveHeight', 'windVelocity10Meter') are
        supported.
    :param output_sampling_interval: desired output sampling rate.

    :return: Pandas DataFrame, containing columns for latitude, longitude, the requested veriables and the satellite
        name.
    """

    # If no satellite list is given assume we want data from all available satellites
    if satellites is None:
        satellites = get_satellite_available()

    # Get the aws keys associated with the request.
    aws_keys = _get_s3_keys(start_date, end_date, satellites)

    output = []
    for satellite in satellites:
        satellite: satellite_formats
        format = _get_format(satellite)

        # Construct inverse mapping from sofar name to netcdf name.
        mapping = format["mapping_variable_name_to_sofar"]
        inv_mapping = {mapping[key]: key for key in mapping}

        # Open netcdf files and load data into individual Pandas Dataframes.
        data = _get_from_cache(
            aws_keys[satellite], output_sampling_interval, inv_mapping, variables
        )

        if data is None:
            continue

        # Get rid of duplicate entries due to overlap in netcdf files
        if not data.index.name == "time":
            data = data.set_index("time")

        data = data.sort_index()
        data = data[~data.index.duplicated(keep="first")]
        data = data.reset_index()

        # add the satellite name as a column.
        data["satellite"] = satellite
        output.append(data)

    # Concatenate individual dataframes for each satellite into a single Dataframe and return.
    if len(output) < 1:
        return None
    else:
        return pandas.concat(output, ignore_index=True)


def get_raw_satellite_data(
    start_date: datetime,
    end_date: datetime,
    satellites: Sequence[SATELLITES] = None,
) -> Dict[str,List[str]]:
    """
    Download satellite data that is stored on s3 for the given interval specificed by start date and end date.

    :param start_date: start date of interval
    :param end_date: end date of interval
    :param satellites: List of satellites to download data from - options are ['jason-3', 'saral', 'sentinal-6'], if
        None is provided as input (default) data from all available satellites is downloaded.
    :param variables: List of Variables to download. Currently only ('significantWaveHeight', 'windVelocity10Meter') are
        supported.
    :param output_sampling_interval: desired output sampling rate.

    :return: Pandas DataFrame, containing columns for latitude, longitude, the requested veriables and the satellite
        name.
    """

    # If no satellite list is given assume we want data from all available satellites
    if satellites is None:
        satellites = get_satellite_available()

    # Get the aws keys associated with the request.
    aws_keys = _get_s3_keys(start_date, end_date, satellites)

    output = {}
    for satellite in satellites:
        # Load data into cache and get filenames
        output[satellite] = filecache.filepaths(aws_keys[satellite])

    return output


def get_satellite_available() -> Sequence[str]:
    return ("jason-3", "saral", "sentinal-6")


def _open_grouped_netcdf_file(file, inv_mapping, variables) -> DataFrame:
    """
    Open a netcdf file and return the requested variables. The netcdf file  (may) contain groups
    (essentially nested datasets).

    :param file: path to netcdf file
    :param inv_mapping: mapping from netcdf variable name to output variable name
    :param variables: variables to return (sofar convention)
    :return: pandas dataframe with requested variables.
    """

    def _open_groups(
        data: netCDF4.Dataset, inv_mapping, variable
    ) -> Union[None, netCDF4.Variable]:
        """
        Function to recursively search netcdf files groups for the desired variables. The first instance of a variable
        that matches the requested variable name is returned. If after traversing all groups no match is found None
        is returned

        :param data: netcdf dataset (or group)
        :param inv_mapping: mapping from sofar name to netcdf name
        :param variable: variable name (sofar convention)
        :return: netcdf variable
        """
        key = inv_mapping[variable]
        if key in data.variables:
            return data.variables[key]

        else:
            groups = data.groups
            if len(groups) > 0:
                for key, group in groups.items():
                    data = _open_groups(group, inv_mapping, variable)
                    if data is not None:
                        return data

        return None

    dataframe = DataFrame()

    # Add the coordinates to the output variables.
    variables = ["time", "latitude", "longitude"] + list(variables)
    with netCDF4.Dataset(file) as dataset:
        # Add variables to dataframe
        for variable in variables:
            # Extract the variable from the netcdf dataset (recursively search groups for desired variable). The
            # satellite data can be stored in a grouped Netcdf format.
            data = _open_groups(dataset, inv_mapping, variable)
            if data is not None:
                # If the variable is time, ensure we convert to a proper time datatype.
                if variable == "time":
                    data = netCDF4.num2date(
                        data[:],
                        data.units,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True,
                    )

                # Store in dataframe, the [:] ensure we return a numpy array from the netcdf variable class.
                dataframe[variable] = data[:]

    # Ensure longitudes are within [-180, 180)
    dataframe["longitude"] = (dataframe["longitude"] + 180) % 360 - 180
    dataframe.set_index("time", inplace=True)
    return dataframe


def _get_format(satellite: SATELLITES) -> Dict:
    return satellite_formats[satellite]


def _bucket() -> str:
    return "sofar-wx-data-dev-os1"


def _get_s3_keys(
    start_date: datetime, end_date: datetime, satellites: Sequence[SATELLITES]
) -> Dict[str, List[str]]:
    """
    Get s3 keys for satellites
    :param start_date: start date
    :param end_date:  end date
    :param satellites: satellites.
    :return: Returns a dictionary with satelite names as key and as values a list of aws keys to the netcdf files that
    contain the data for that satellite withing the requested range.
    """

    bucket = boto3.resource("s3").Bucket(_bucket())

    # Construct the dat/cycles we are interested in. We use the timeslice logic here as the satellite data is stored
    # on 6 hour intervals - similar to the forecast cycles.
    time_slice = TimeSliceLead(
        start_date, end_date, timedelta(hours=0), endpoint=True, exact=True
    )

    satelite_prefixes = {
        satellite: _get_format(satellite)["prefix"] for satellite in satellites
    }

    keys = {satellite: [] for satellite in satellites}

    # Unfortunately we cannot programatically construct the keys (due to naming on s3). Instead we have to enquire to
    # contents of a folder.
    for date in time_slice.time_base(ModelTimeConfiguration()):

        # build the prefix
        time = date[0] + date[1]
        date_string = time.strftime("%Y%m%d")
        hour_string = time.strftime("%H")
        prefix = f"observations/raw-files/data-assimilation/satellite-altimeter/{date_string}/{hour_string}"

        # Loop over all objects that match the prefix
        for object in bucket.objects.filter(Prefix=prefix):

            # loop over satelites - if the satellite prefix matches - add the object to the list for that satellite.
            for satellite, prefix in satelite_prefixes.items():
                if prefix in object.key:
                    keys[satellite].append(f"s3://{_bucket()}/{object.key}")
                    break
    return keys


def _resample_dataframe(dataframe: DataFrame, output_sampling_interval) -> DataFrame:
    """
    Resample a dataframe ensuring that longitude is properly accounted for. Data must be sorted by time for this to
    work.
    :param dataframe:
    :param output_sampling_interval:
    :return:
    """
    if dataframe.index.name == "time":
        df = dataframe
    else:
        df = dataframe.set_index("time")

    if "longitude" in df:
        df["longitude"] = numpy.unwrap(df["longitude"], discont=180, period=360)

    # Resample the data to desired interval and take the average value to represent the bin
    df = df.resample(output_sampling_interval).mean().reset_index()

    # We may introduce NaN's when resampling over gaps in the data- lets drop those rows.
    df = df.dropna()

    if "longitude" in df:
        df["longitude"] = (df["longitude"] + 180) % 360 - 180
    return df
