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
from roguewave.filecache import filepaths
import netCDF4
import pandas


SATELLITES = Literal["jason-3", "saral", "sentinal-6"]


satellite_formats = {
    "jason-3": {
        "prefix": "JA3",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windSpeed",
            "swh_ocean": "significantWaveHeight",
            "time": "time",
            "latitude": "latitude",
            "longitude": "longitude",
        },
    },
    "sentinal-6": {
        "prefix": "S6A",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windSpeed",
            "swh_ocean": "significantWaveHeight",
            "time": "time",
            "latitude": "latitude",
            "longitude": "longitude",
        },
    },
    "saral": {
        "prefix": "SRL",
        "mapping_variable_name_to_sofar": {
            "wind_speed_alt": "windSpeed",
            "swh": "significantWaveHeight",
            "time": "time",
            "lat": "latitude",
            "lon": "longitude",
        },
    },
}


def get_satellite_data(
    start_date: datetime,
    end_date: datetime,
    satellites: Sequence[SATELLITES] = None,
    variables=("significantWaveHeight", "windSpeed"),
) -> DataFrame:
    """
    Download satellite data that is stored on s3 for the given interval specificed by start date and end date.

    :param start_date: start date of interval
    :param end_date: end date of intervak
    :param satellites: List of satellites to download data from - options are ['jason-3', 'saral', 'sentinal-6'], if
        None is provided as input (default) data from all available satellites is downloaded.
    :param variables: List of Variables to download. Currently only ('significantWaveHeight', 'windSpeed') are
        supported.

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
        _data = []

        # Download and store s3 object locally in cache (if not already available), and return paths to the
        # local copies.
        files = filepaths(aws_keys[satellite])
        for file in files:
            _data.append(_open_grouped_netcdf_file(file, inv_mapping, variables))

        # Concatenate individual data frames into a single dataframe for the satellite, and add the
        # satellite name as a column.
        _data = pandas.concat(_data, ignore_index=True)
        _data["satellite"] = satellite
        output.append(_data)

    # Concatenate individual dataframes for each satellite into a single Dataframe and return.
    return pandas.concat(output, ignore_index=True)


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
