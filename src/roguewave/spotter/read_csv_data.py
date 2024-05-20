"""
Contents: Routines to read raw data from Sofar Spotters

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines read csv data stored on SD-Cards and return in a convenient form.

Public Functions:

- `read_baro`, read ????_BARO.csv files and return a dataframe.
- `read_baro_raw`, read ????_BARO_RAW.csv files and return a dataframe.
- `read_gmn`, read ????_GMN.csv files and return a dataframe.
- `read_raindb`, read ????_RAINDB.csv files and return a dataframe.
- `read_sst`, read ????_SST.csv files and return a dataframe.
- `read_displacement`, read ????_FLT.csv files that contain displacement data
- `read_gps`, read ????_GPS.csv files that contain raw GPS strings.
- `read_location`, read ????_LOC.csv files that containt the location if the instrument.
- `read_raw_spectra`, read ????_SPC.csv files that contain raw spectral data.
- `read_spectra`, read ????_SPC.csv files and return a spectral object.
- `read_rbrdt`, read ????_smd.csv files and return a dataframe.


"""
import typing

import numpy
import pandas as pd
from pandas import DataFrame, concat, to_datetime

from roguewave.spotter.parser import (
    read_and_concatenate_spotter_csv,
    get_csv_file_format,
    apply_to_group,
    _mark_continuous_groups,
    save_as_netcdf,
    _netcdf_filename
)
from roguewave.timeseries_analysis.filtering import sos_filter
from xarray import Dataset
from roguewave import FrequencySpectrum
from roguewave.tools.time import datetime64_to_timestamp
from datetime import datetime
from numpy import linspace, errstate, sqrt, interp, full_like, inf, cos, sin, pi, nan
import xarray
import os
import typing

# Main Functions
# ---------------------------------


def read_gps(
    path,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess=True,
    cache_as_netcdf=False,
) -> DataFrame:
    """
    Load raw GPS text files and return a pandas dataframe containing the data. By default the data is postprocessed into
    a more  convinient form (without loss of information) unless raw data is specifically requested

    :param path: Path containing Spotter CSV files
    :param postprocess: whether to postprocess the data. Postprocessing converts heading and velocity magnitude to
                        velocity components, and combines latitude and lituted minutes into a single double latitude
                        (same for longitudes).

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

    data = read_and_concatenate_spotter_csv(
        path,
        "GPS",
        start_date=start_date,
        end_date=end_date,
        cache_as_netcdf=cache_as_netcdf,
    )
    if not postprocess:
        return data
    u = (
        data["SOG(mm_s)"].values
        / 1000
        * cos((90 - data["COG(deg*1000)"].values / 1000) * pi / 180)
    )
    v = (
        data["SOG(mm_s)"].values
        / 1000
        * sin((90 - data["COG(deg*1000)"].values / 1000) * pi / 180)
    )
    w = data["vert_vel(mm_s)"].values / 1000
    time = to_datetime(data["GPS_Epoch_Time(s)"].values, unit="s")

    latitude = data["lat(deg)"].values + data["lat(min*1e5)"].values / 60e5
    longitude = data["long(deg)"].values + data["long(min*1e5)"].values / 60e5
    z = data["el(m)"].values

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
    cache_as_netcdf=False,
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

    :return: Pandas Dataframe. Returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "x": filteresd displacement data (Eastings)
             "y': filteresd displacement data (Northings)
             "z': vertical displacement from local mean.
             "group id": Identifier that indicates continuous data groups.
                         (data from "same deployment).
    """

    data = read_and_concatenate_spotter_csv(
        path,
        "FLT",
        start_date=start_date,
        end_date=end_date,
        cache_as_netcdf=cache_as_netcdf,
    )
    if not postprocess:
        return data

    # Backward sos pass for phase correction
    def _process(_data: DataFrame):
        _df = DataFrame()
        _df["time"] = _data["time"]
        _df["x"] = sos_filter(_data["outx(mm)"].values / 1000.0, "backward")
        _df["y"] = sos_filter(_data["outy(mm)"].values / 1000.0, "backward")
        _df["z"] = sos_filter(_data["outz(mm)"].values / 1000.0, "backward")
        return _df

    return apply_to_group(_process, data)


def read_baro(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = False,
):
    """
    Read filtered barometer files.

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
    """
    raw_data = read_and_concatenate_spotter_csv(path, "BARO", start_date, end_date)
    if not postprocess:
        return raw_data

    data = DataFrame()
    data["time"] = raw_data["time"]
    data["pressure (pascal)"] = raw_data["pressure (mbar)"] * 100
    return data



def read_baro_raw(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = True,
):
    """
    Read raw barometer files (no filtering)

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
    """
    raw_data = read_and_concatenate_spotter_csv(path, "BARO_RAW", start_date, end_date)
    if not postprocess:
        return raw_data

    data = DataFrame()
    data["time"] = raw_data["time"]
    data["pressure (pascal)"] = raw_data["pressure (mbar)"] * 100
    return data


def read_sst(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
):
    """
    Read SST data.

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
    """
    return read_and_concatenate_spotter_csv(path, "sst", start_date, end_date)


def read_raindb(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
):
    """
    Read sound volume in decibel.

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
    """
    return read_and_concatenate_spotter_csv(path, "RAINDB", start_date, end_date)


def read_gmn(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
):
    """
    Read gps metric files- for development purposes only.

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
    """
    return read_and_concatenate_spotter_csv(path, "GMN", start_date, end_date)


def read_location(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = True,
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

    :return: Pandas Dataframe. Returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "latitude": latitude in decimal degrees
             "longitude': longitude in decimal degrees
             "group id": Identifier that indicates continuous data groups.
                         (data from "same deployment).
    """
    raw_data = read_and_concatenate_spotter_csv(
        path,
        "LOC",
        start_date=start_date,
        end_date=end_date,
    )
    if not postprocess:
        return raw_data

    dataframe = DataFrame()
    dataframe["time"] = raw_data["time"]

    dataframe["latitude"] = (
        raw_data["lat(deg)"].values + raw_data["lat(min*1e5)"].values / 60e5
    )
    dataframe["longitude"] = (
        raw_data["long(deg)"].values + raw_data["long(min*1e5)"].values / 60e5
    )
    dataframe["group id"] = raw_data["group id"]
    return dataframe


def read_spectra(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    depth: float = inf,
    cache_as_netcdf=False,
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

    :return: frequency spectra as a FrequencySpectrum object.
    """

    data = read_raw_spectra(
        path, start_date=start_date, end_date=end_date, cache_as_netcdf=cache_as_netcdf
    )
    spectral_file_format = get_csv_file_format("SPC")
    df = 1 / (
        spectral_file_format["nfft"] * spectral_file_format["sampling_interval_gps"]
    )
    frequencies = (
        linspace(
            0,
            spectral_file_format["nfft"] // 2,
            spectral_file_format["nfft"] // 2,
            endpoint=False,
        )
        * df
    )

    spectral_values = data[list(range(0, spectral_file_format["nfft"] // 2))].values
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
        location = read_location(path, postprocess=True)
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
    start_date: datetime = None,
    end_date: datetime = None,
    cache_as_netcdf=False,
) -> DataFrame:
    """
    Read raw spectral files and return a dataframe

    :param path:
    :param postprocess:
    :param start_date:
    :param end_date:
    :return:
    """
    raw_data = read_and_concatenate_spotter_csv(
        path,
        "SPC",
        start_date=start_date,
        end_date=end_date,
        cache_as_netcdf=cache_as_netcdf,
    )
    if not postprocess:
        return raw_data

    # We want to collect all spectra of the same type (Sxx_re or Syy_im) in a single dataframe, with frequencies as
    # rows. First, we list all of the columns associated with a particular variable.
    spec_format = get_csv_file_format("SPC")
    column_names = {}

    # Loop over the columns in the spec format
    for column in spec_format["columns"]:
        # If it is not one of the spectral columns, continue (e.g. time)
        if column["name"][0:3] not in ["Sxx", "Syy", "Szz", "Sxy", "Szx", "Szy"]:
            continue

        # Get the full name of the column (Sxx_re, or Sxy_im) without the frequency enumeration. This is the variable
        # of interest
        name = column["name"][0:6]

        # add full column name to the list of columns belonging to this variable. e.g. we get for
        # {
        #   "Szz_re": ["Szz_re_0", "Szz_re_1", ..., "Szz_re_128"],
        #   ....         ....         ....     ....       ....
        #   "Szy_re": ["Szy_im_0", "Szy_im_1", ..., "Szy_im_128"]
        # }
        if name not in column_names:
            column_names[name] = []

        column_names[name].append(column["name"])

    # initialize
    dataframes = []
    data_types = []

    frequency_resolution = 1 / (
        spec_format["sampling_interval_gps"] * spec_format["nfft"]
    )

    # Construct a dataframe for each variable and add to the list of dataframes.
    for data_type, freq_column_names in column_names.items():
        df = raw_data.loc[:, ["time"] + freq_column_names]
        df.loc[:, freq_column_names] = df[freq_column_names] / (
            1000000.0 * frequency_resolution
        )

        rename_mapping = {name: index for index, name in enumerate(freq_column_names)}
        df = df.rename(rename_mapping, axis=1)
        data_types.append(data_type)
        dataframes.append(df)

    data = concat(dataframes, keys=data_types, names=["kind", "source_index"])
    data.reset_index(inplace=True)
    data.drop(columns="source_index", inplace=True)
    return data

def read_rbr(
    path: str,
    start_date: datetime = None,
    end_date: datetime = None,
    postprocess: bool = True,
    sampling_interval_seconds: float = 0.5,
    sensor_type:typing.Literal['RBRD','RBRDT'] = 'RBRD',
    cache_as_netcdf=False,
    **kwargs
) -> DataFrame:
    """
    Read RBR data from csv files and return a dataframe containing the data.

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

    :param postprocess: whether to remove data points with no GPS fix, and to convert from microbar to pascal.

    :param sampling_interval_seconds: The sampling interval of the data in seconds. This is used to determine whether
                                      data points are continuous or not.

    :param sensor_type: The type of sensor to read. Must be either "RBRD" or "RBRDT".
        "RBRD" is the standard RBR sensor, which only measures pressure.
        "RBRDT" is the RBR sensor with a temperature sensor, which measures pressure and temperature.

    :param cache_as_netcdf: If True, the data is saved as a netcdf file in the same directory as the csv files. If the
                            netcdf file already exists, the data is read from the netcdf file instead of the csv files.

    :return: Pandas Dataframe. Returns dataframe with columns

             "time": epoch time (UTC, epoch of 1970-1-1, i.e. Unix Epoch).
             "pressure (pascal)": Pressure in pascal
             "group id": Identifier that indicates continuous data groups (data without gaps).

    """



    correct_for_atmospheric_pressure = kwargs.get("correct_for_atmospheric_pressure", True)
    mean_pressure_pa = kwargs.get("mean_pressure_pa", 101325)
    use_barometer = kwargs.get("use_barometer", True)

    if not sensor_type in ['RBRD','RBRDT']:
        raise ValueError('Invalid type, must be either "RBRD" or "RBRDT"')

    if cache_as_netcdf and os.path.exists(_netcdf_filename(path, sensor_type)):
        raw_data = xarray.open_dataset(_netcdf_filename(path, sensor_type)).to_dataframe()
    else:
        raw_data = read_and_concatenate_spotter_csv(
            path,
            sensor_type,
            start_date=start_date,
            end_date=end_date,
        )

    # Only keep the data from the smartmooring file for the sensor type of interest
    raw_data = raw_data[
        raw_data['sensor'] == sensor_type
    ]

    # save as netcdf if requested
    if cache_as_netcdf:
        save_as_netcdf(raw_data, path, sensor_type)

    if not postprocess:
        return raw_data

    # If the gps does not yet have a fix, the time is set to 1970-01-01. We remove these data points. The filter date
    # is arbitrary- but since no smartmoorings existed before 2020, we can safely remove all data before 2020.
    raw_data = raw_data[
        raw_data['time'] > pd.Timestamp('1990-01-01')
    ]

    # Output processed data in a dataframe.
    dataframe = DataFrame()
    dataframe["time"] = raw_data["time"]
    dataframe["pressure (pascal)"] = raw_data["pressure (microbar)"] / 10
    dataframe = _mark_continuous_groups(dataframe, sampling_interval_seconds)

    if correct_for_atmospheric_pressure:
        interpolated_pressure = mean_pressure_pa
        if use_barometer:
            baro = read_baro(path,postprocess=True)
            baro_pressure = baro["pressure (pascal)"].values.astype(float)
            baro_time = baro["time"].values.astype(float)
            rbr_time = dataframe["time"].values.astype(float)
            interpolated_pressure = interp(rbr_time, baro_time, baro_pressure)
        else:
            pass

        dataframe["pressure (pascal)"] = dataframe["pressure (pascal)"] - interpolated_pressure

    return dataframe