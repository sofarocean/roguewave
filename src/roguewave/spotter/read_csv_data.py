"""
Contents: Routines to read raw data from Sofar Spotters

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines read csv data stored on SD-Cards and return in a convenient form.

Public Functions:

- `read_displacement`, read ????_FLT.csv files that contain displacement data
- `read_gps`, read ????_GPS.csv files that contain raw GPS strings.
- `location`, read ????_LOC.csv files that containt the location if the instrument.
- `read_raw_spectra`, read ????_SPC.csv files that contain raw spectral data.
- `read_spectra`, read ????_SPC.csv files and return a spectral object.
"""

from ._csv_file_layouts import (
    get_format,
    ColumnParseParameters,
    spectral_column_names,
    file_name_pattern,
)
from ._spotter_constants import spotter_constants, SpotterConstants
from pandas import DataFrame, read_csv, concat, to_datetime, to_numeric
from roguewave.timeseries_analysis.filtering import sos_filter
from typing import Iterator, List, Callable
from xarray import Dataset
from glob import glob
from roguewave import FrequencySpectrum
from roguewave.tools.time import datetime64_to_timestamp, to_datetime_utc, to_datetime64
from datetime import datetime
import os
from numpy import (
    linspace,
    errstate,
    sqrt,
    interp,
    full_like,
    inf,
    cos,
    sin,
    pi,
    timedelta64,
    all,
    nan,
)


# Main Functions
# ---------------------------------


def read_gps(
    path,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess=True,
    config: SpotterConstants = None,
) -> DataFrame:
    """
    Load raw GPS text files and return a pandas dataframe containing the data. By default the data is postprocessed into
    a more  convinient form (without loss of information) unless raw data is specifically requested

    :param path: Path containing Spotter CSV files
    :param postprocess: whether to postprocess the data. Postprocessing converts heading and velocity magnitude to
                        velocity components, and combines latitude and lituted minutes into a single double latitude
                        (same for longitudes).

    :param config: set of default settings in the spotter processing pipeline. Only needed for development purposes.

    :param start_date: If only a subset of the data is needed we can avoid loading all data, this denotes the start
                       date of the desired interval. If given, only data after the start_date is loaded (if available).
                       NOTE: this requires that LOC files are present.

    :param end_date: If only a subset of the data is needed we can avoid loading all data, this denotes the end
                       date of the desired interval. If given, only data before the end_date is loaded (if available).
                       NOTE: this requires that LOC files are present.

    :return: Pandas Dataframe. If postprocess is false it just contains the raw columns of the GPS file (see file for
             description) If True (default) returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "latitude": Latitude in decimal degrees
             "longitude": Longitude in decimal degrees
             "z": raw vertical elevation from GPS in meter
             "u': eastward velocity, m/s
             "v': northward velocity, m/s
             "w": vertical velocity, m/s
             "group id": Identifier that indicates continuous data groups. (data from "same deployment).

    """

    if config is None:
        config = spotter_constants()

    data = _read_data(
        path,
        "GPS",
        sampling_interval=config["sampling_interval_gps"],
        start_date=start_date,
        end_date=end_date,
    )
    if not postprocess:
        return data
    u = data["speed over ground"].values * cos(
        (90 - data["course over ground"].values) * pi / 180
    )
    v = data["speed over ground"].values * sin(
        (90 - data["course over ground"].values) * pi / 180
    )
    w = data["w"].values
    time = data["time"].values

    latitude = data["latitude degrees"].values + data["latitude minutes"].values / 60
    longitude = data["longitude degrees"].values + data["longitude minutes"].values / 60
    z = data["z"].values

    return DataFrame(
        data={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
            "z": z,
            "u": u,
            "v": v,
            "w": w,
            "group id": data["group id"].values,
        }
    )


def read_displacement(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = True,
    config: SpotterConstants = None,
) -> DataFrame:

    """
    Load displacement dataand return a pandas dataframe containing the data.
    By default the data is postprocessed to apply an inverse pass of the
    IIR filter to correct for phase differences.

    :param path: Path containing Spotter CSV files

    :param start_date: If only a subset of the data is needed we can avoid
                       loading all data, this denotes the start date of the
                       desired interval. If given, only data after the
                       start_date is loaded (if available).
                       NOTE: this requires that LOC files are present.

    :param end_date: If only a subset of the data is needed we can avoid
                     loading all data, this denotes the end date of the
                     desired interval. If given, only data before the
                     end_date is loaded (if available).
                     NOTE: this requires that LOC files are present.

    :param postprocess: whether to apply the phase correction

    :param config: set of default settings in the spotter processing
                   pipeline. Only needed for development purposes.

    :return: Pandas Dataframe. Returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "x": filteresd displacement data (Eastings)
             "y': filteresd displacement data (Northings)
             "z': vertical displacement from local mean.
             "group id": Identifier that indicates continuous data groups.
                         (data from "same deployment).
    """

    if config is None:
        config = spotter_constants()

    data = _read_data(
        path,
        "FLT",
        sampling_interval=config["sampling_interval_location"],
        start_date=start_date,
        end_date=end_date,
    )
    if not postprocess:
        return data

    # Backward sos pass for phase correction
    def _process(_data: DataFrame):
        _data["x"] = sos_filter(_data["x"].values, "backward")
        _data["y"] = sos_filter(_data["y"].values, "backward")
        _data["z"] = sos_filter(_data["z"].values, "backward")
        return _data

    return apply_to_group(_process, data)


def read_location(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = True,
    config: SpotterConstants = None,
) -> DataFrame:
    """

    :param path: Path containing Spotter CSV files

    :param start_date: If only a subset of the data is needed we can avoid
                       loading all data, this denotes the start date of the
                       desired interval. If given, only data after the
                       start_date is loaded (if available).
                       NOTE: this requires that LOC files are present.

    :param end_date: If only a subset of the data is needed we can avoid
                     loading all data, this denotes the end date of the
                     desired interval. If given, only data before the
                     end_date is loaded (if available).
                     NOTE: this requires that LOC files are present.

    :param postprocess: whether to apply the phase correction

    :param config: set of default settings in the spotter processing
                   pipeline. Only needed for development purposes.

    :return: Pandas Dataframe. Returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "latitude": latitude in decimal degrees
             "longitude': longitude in decimal degrees
             "group id": Identifier that indicates continuous data groups.
                         (data from "same deployment).
    """

    if config is None:
        config = spotter_constants()

    raw_data = _read_data(
        path,
        "LOC",
        config["sampling_interval_location"],
        start_date=start_date,
        end_date=end_date,
    )
    if not postprocess:
        return raw_data

    dataframe = DataFrame()
    dataframe["time"] = raw_data["time"]
    dataframe["latitude"] = (
        raw_data["latitude degrees"] + raw_data["latitude minutes"] / 60
    )
    dataframe["longitude"] = (
        raw_data["longitude degrees"] + raw_data["longitude minutes"] / 60
    )
    dataframe["group id"] = raw_data["group id"]
    return dataframe


def read_spectra(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    depth: float = inf,
    config: SpotterConstants = None,
) -> FrequencySpectrum:
    """
    Read spectral data from csv files. The raw spectral data is transformed into
    a roguewave Spectral 1D spectrum object (which includes all directional moments a1,b1,a2,b2 as well as energy for
    the given time period).

    :param path: Path containing Spotter CSV files
    :param postprocess: Whether to apply the phase correction

    :param start_date: If only a subset of the data is needed we can avoid
                       loading all data, this denotes the start date of the
                       desired interval. If given, only data after the
                       start_date is loaded (if available).
                       NOTE: this requires that LOC files are present.

    :param end_date: If only a subset of the data is needed we can avoid
                     loading all data, this denotes the end date of the
                     desired interval. If given, only data before the
                     end_date is loaded (if available).
                     NOTE: this requires that LOC files are present.

    :param depth: Local water depth,  by default set to inf (deep water).
                  Not required, but is set on the returned spectral object
                  (and factors in transformations thereof, e.g. to get wavenumbers).

    :param config: set of default settings in the spotter processing
                   pipeline. Only needed for development purposes.

    :return: frequency spectra as a FrequencySpectrum object.
    """

    if config is None:
        config = spotter_constants()

    data = read_raw_spectra(path, start_date=start_date, end_date=end_date)
    sampling_interval = config["sampling_interval_gps"] / timedelta64(1, "s")
    df = 1 / (config["number_of_samples"] * sampling_interval)
    frequencies = (
        linspace(
            0,
            config["number_of_frequencies"],
            config["number_of_frequencies"],
            endpoint=False,
        )
        * df
    )

    spectral_values = data[list(range(0, config["number_of_frequencies"]))].values
    time = data["time"].values
    time = time[data["kind"] == "Szz_re"]

    Szz = spectral_values[data["kind"] == "Szz_re", :]
    Sxx = spectral_values[data["kind"] == "Sxx_re", :]
    Syy = spectral_values[data["kind"] == "Syy_re", :]
    Cxy = spectral_values[data["kind"] == "Sxy_re", :]
    Qzx = spectral_values[data["kind"] == "Szx_im", :]
    Qzy = spectral_values[data["kind"] == "Szy_im", :]

    with errstate(invalid="ignore", divide="ignore"):
        # Supress divide by 0; silently produce NaN
        a1 = Qzx / sqrt(Szz * (Sxx + Syy))
        a2 = (Sxx - Syy) / (Sxx + Syy)
        b1 = Qzy / sqrt(Szz * (Sxx + Syy))
        b2 = 2.0 * Cxy / (Sxx + Syy)

    try:
        location = read_location(path, postprocess=True, config=config)
        latitude = interp(
            datetime64_to_timestamp(time),
            datetime64_to_timestamp(location["time"].values),
            location["latitude"].values,
        )
        longitude = interp(
            datetime64_to_timestamp(time),
            datetime64_to_timestamp(location["time"].values),
            location["longitude"].values,
        )

    except FileNotFoundError:
        # No location files find to get latitude/longitude. Just fill with NaN.
        latitude = full_like(time, nan)
        longitude = full_like(time, nan)

    depth = full_like(time, depth)

    dataset = Dataset(
        data_vars={
            "variance_density": (["time", "frequency"], Szz),
            "a1": (["time", "frequency"], a1),
            "b1": (["time", "frequency"], b1),
            "a2": (["time", "frequency"], a2),
            "b2": (["time", "frequency"], b2),
            "depth": (["time"], depth),
            "latitude": (["time"], latitude),
            "longitude": (["time"], longitude),
        },
        coords={"time": to_datetime(time), "frequency": frequencies},
    )
    return FrequencySpectrum(dataset)


def read_raw_spectra(
    path,
    postprocess=True,
    config: SpotterConstants = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    if config is None:
        config = spotter_constants()
    if not postprocess:
        return _read_data(
            path,
            "SPC",
            config["sampling_interval_spectra"],
            start_date=start_date,
            end_date=end_date,
        )

    if config is None:
        config = spotter_constants()

    column_names = spectral_column_names(groupedby="kind", config=config)

    dataframes = []
    raw_data = _read_data(path, "SPC", config["sampling_interval_spectra"])

    data_types = []
    for data_type, freq_column_names in column_names:
        df = raw_data[["time"] + freq_column_names]
        rename_mapping = {name: index for index, name in enumerate(freq_column_names)}
        df = df.rename(rename_mapping, axis=1)
        data_types.append(data_type)
        dataframes.append(df)

    data = concat(dataframes, keys=data_types, names=["kind", "source_index"])
    data.reset_index(inplace=True)
    data.drop(columns="source_index", inplace=True)
    return data


# Private Functions
# ---------------------------------


def _files_to_parse(
    path: str, pattern: str, start_date: datetime = None, end_date: datetime = None
) -> Iterator[str]:
    """
    Create a generator that yields files that conform to a given filename pattern and that contain data within the
    requested interval. Returns an iterator.

    :param path: path containing Spotter data files
    :param pattern: file pattern to look for
    :param start_date: start date of the interval, if none no lower interval is imposed.
    :param end_date: end date of the interval, if none no upper interval is imposed.
    :return: generator that yields files qith data conforming to the request.
    """

    # Get the files conforming to a specific pattern- say ????_LOC.csv
    # For Spotter these are the location files, 0000_LOC.csv, 0001_LOC.csv, etc.
    files = glob(os.path.join(path, pattern))

    if (start_date is not None) or (end_date is not None):
        # If we are only interested in a range of data we do not want to parse all the text files. Specifically the
        # location, gps and spectral files can get very large (hundreds of mb) and consequenlty slow to read. Spotter
        # saves this data onboard split across multiple files (e.g. 0000_FLT.csv, 0001_FLT.csv etc) so we could just
        # load only those files containing the time range we need. We do not in general know which files contain which
        # time range though. Here we there try to read the timebase from the smallest data files (location files)
        # and use that to determine which files we read.

        # Read the files containing the timebase.
        time_files = glob(os.path.join(path, file_name_pattern["TIME"]))
        csv_parsing_options = get_format("TIME")

        if not (len(time_files) == len(files)):
            # We need an equal number of files - otherwise something is amiss
            raise ValueError(
                "Number of location files needs to be the same as the number of files being read to use"
                "time filtering"
            )

        # For each data file and time base file do
        for file, time_file in zip(sorted(files), sorted(time_files)):

            # Read the time base file to get the time range in the XXXX_VAR.csv file
            time = read_csv(
                time_file,
                index_col=False,
                delimiter=",",
                header=0,
                names=[x["column_name"] for x in csv_parsing_options],
                dtype={x["column_name"]: x["dtype"] for x in csv_parsing_options},
                on_bad_lines="skip",
                usecols=[x["column_name"] for x in csv_parsing_options if x["include"]],
            )
            file_in_range = False
            if time.shape[0] == 0:
                # If empty- continue
                continue

            # If a start date is given- check if any data in the file has a timestamp after the start date
            if start_date is not None:
                start_date_timestamp = to_datetime_utc(start_date).timestamp()
                file_in_range = not (
                    start_date_timestamp > time["time"].values[-1] + 60.0
                )

            # If an end date is given- check if any data in the file has a timestamp before the end date
            if end_date is not None:
                end_date_timestamp = to_datetime_utc(end_date).timestamp()
                file_in_range = file_in_range and not (
                    end_date_timestamp <= time["time"].values[0]
                )

            if file_in_range:
                yield file
            else:
                continue

    else:
        for file in sorted(files):
            yield file


def _load_as_dataframe(
    files_to_parse: Iterator[str],
    csv_parsing_options: List[ColumnParseParameters],
    sampling_interval=0.4,
) -> DataFrame:

    """
    Main data loading function.
    :param files_to_parse:
    :param csv_parsing_options:
    :param sampling_interval:
    :return:
    """

    column_names = [x["column_name"] for x in csv_parsing_options]
    usecols = [x["column_name"] for x in csv_parsing_options if x["include"]]
    dtype = {x["column_name"]: x["dtype"] for x in csv_parsing_options}
    convert = [x["convert"] for x in csv_parsing_options if x["include"]]

    def process_file(file):
        df = read_csv(
            file,
            index_col=False,
            delimiter=",",
            header=0,
            names=column_names,
            on_bad_lines="skip",
            usecols=usecols,
            low_memory=False,
        )

        for key in usecols:
            if dtype[key] == str:
                continue
            df[key] = to_numeric(df[key], errors="coerce")

        df = _quality_control(df)

        if "time" in df:
            df["time"] = to_datetime(df["time"], unit="s")

        for name, function in zip(usecols, convert):
            df[name] = function(df[name])

        return df

    source_files = list(files_to_parse)
    if len(source_files) == 0:
        raise FileNotFoundError("No files to parse")

    data_frames = [process_file(source_file) for source_file in source_files]

    # Fragmentation occurs for spectral data- to avoid performance issues we recreate the dataframe after the concat
    # here.
    dataframe = concat(
        data_frames, keys=source_files, names=["source files", "file index"]
    ).copy()
    dataframe.reset_index(inplace=True)
    return _mark_continuous_groups(dataframe, sampling_interval)


def _read_data(
    path,
    data_type,
    sampling_interval,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    files = _files_to_parse(path, file_name_pattern[data_type], start_date, end_date)

    format = get_format(data_type)
    dataframe = _load_as_dataframe(files, format, sampling_interval)

    if start_date is not None:
        start_date = to_datetime64(start_date)
        dataframe = dataframe[dataframe["time"] >= start_date]

    if end_date is not None:
        end_date = to_datetime64(end_date)
        dataframe = dataframe[dataframe["time"] < end_date]

    return dataframe


# Utility Functions
# ---------------------------------


def _quality_control(dataframe: DataFrame) -> DataFrame:
    # Remove any rows with nan entries
    dataframe = dataframe.dropna()

    # Here we check for negative intervals and only keep data with positive intervals. It needs to be recursively
    # applied since when removing an interval we may introduce a new negative interval. There are definitely more
    # efficient ways to implement this and if this ever is a bottleneck we should - until then we do it the ugly way :-)
    while True:
        positive_time = dataframe["time"].diff() > 0.0
        if all(positive_time.values[1:]):
            break
        else:
            dataframe = dataframe.loc[positive_time]

    return dataframe


def _mark_continuous_groups(df: DataFrame, sampling_interval):
    """
    This function adds a column that has unique number for each continuous block of data (i.e. data without gaps).
    It does so by:
        - calculating the time difference
        - creating a boolean mask where the time difference is larger than a threshold
        - taking the cumulative sum over that mask- this sum mask will only increase if there is a gap between succesive
          entries - and can thus serve as a group marker.
    :param df: data frame contaning a time epoch key
    :param sampling_interval: sampling interval used
    :return:
    """

    df["group id"] = (
        df["time"].diff() > sampling_interval + timedelta64(100, "ms")
    ).cumsum()
    return df


def apply_to_group(function: Callable[[DataFrame], DataFrame], dataframe: DataFrame):
    """
    Apply a function to each group seperately and recombine the result into a single dataframe

    :param function: Function to appply
    :param dataframe: Dataframe to apply function to
    :return:
    """
    groups = dataframe.groupby("group id")
    dataframes = [function(x[1]) for x in groups]
    return concat(dataframes)
