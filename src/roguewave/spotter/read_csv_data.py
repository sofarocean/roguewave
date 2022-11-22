from ._csv_file_layouts import get_format, ColumnParseParameters, spectral_column_names
from ._spotter_constants import spotter_constants, SpotterConstants
from pandas import DataFrame, read_csv, concat, to_datetime
from roguewave.timeseries_analysis.filtering import sos_filter
from typing import Iterator, List, Callable
from xarray import Dataset
from glob import glob
from roguewave import FrequencySpectrum
from roguewave.tools.time import datetime64_to_timestamp, to_datetime_utc, to_datetime64
from datetime import datetime
import os
from numpy import (
    nan,
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
)

_pattern = {
    "GPS": "????_GPS.csv",
    "FLT": "????_FLT.csv",
    "LOC": "????_LOC.csv",
    "SPC": "????_SPC.csv",
    "TIME": "????_LOC.csv",
}


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


def files_to_parse(
    path: str, pattern: str, start_date: datetime = None, end_date: datetime = None
) -> Iterator[str]:

    files = glob(os.path.join(path, pattern))

    if (start_date is not None) or (end_date is not None):
        time_files = glob(os.path.join(path, _pattern["TIME"]))
        csv_parsing_options = get_format("TIME")
        if not (len(time_files) == len(files)):
            raise ValueError(
                "Number of location files needs to be the same as the number of files being read to use"
                "time filtering"
            )

        for file, time_file in zip(sorted(files), sorted(time_files)):
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
                continue

            if start_date is not None:
                start_date_timestamp = to_datetime_utc(start_date).timestamp()
                file_in_range = not (
                    start_date_timestamp > time["time"].values[-1] + 60.0
                )

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


def load_as_dataframe(
    files_to_parse: Iterator[str],
    csv_parsing_options: List[ColumnParseParameters],
    sampling_interval=0.4,
) -> DataFrame:
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
            dtype=dtype,
            on_bad_lines="skip",
            usecols=usecols,
        )
        df = quality_control(df)

        if "time" in df:
            df["time"] = to_datetime(df["time"], unit="s")

        for name, function in zip(usecols, convert):
            df[name] = function(df[name])

        return df

    source_files = list(files_to_parse)
    data_frames = [process_file(source_file) for source_file in source_files]

    # Fragmentation occurs for spectral data- to avoid performance issues we recreate the dataframe after the concat
    # here.
    dataframe = concat(
        data_frames, keys=source_files, names=["source files", "file index"]
    ).copy()
    dataframe.reset_index(inplace=True)
    return mark_continuous_groups(dataframe, sampling_interval)


def mark_continuous_groups(df: DataFrame, sampling_interval):
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


def read_data(
    path,
    data_type,
    sampling_interval,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    files = files_to_parse(path, _pattern[data_type], start_date, end_date)
    format = get_format(data_type)
    dataframe = load_as_dataframe(files, format, sampling_interval)

    if start_date is not None:
        start_date = to_datetime64(start_date)
        dataframe = dataframe[dataframe["time"] >= start_date]

    if end_date is not None:
        end_date = to_datetime64(end_date)
        dataframe = dataframe[dataframe["time"] < end_date]

    return dataframe


def read_gps(
    path,
    postprocess=True,
    config: SpotterConstants = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    if config is None:
        config = spotter_constants()

    data = read_data(
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
    path,
    postprocess=True,
    config: SpotterConstants = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    if config is None:
        config = spotter_constants()

    data = read_data(
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
    path,
    postprocess=True,
    config: SpotterConstants = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DataFrame:
    if config is None:
        config = spotter_constants()

    raw_data = read_data(
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
        return read_data(
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
    raw_data = read_data(path, "SPC", config["sampling_interval_spectra"])

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


def read_spectra(
    path,
    depth=inf,
    config: SpotterConstants = None,
    start_date: datetime = None,
    end_date: datetime = None,
) -> FrequencySpectrum:
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


def quality_control(dataframe: DataFrame) -> DataFrame:
    negative_time = dataframe["time"].diff() < 0.0
    dataframe.loc[negative_time, "time"] = nan
    dataframe.dropna(inplace=True)
    return dataframe
