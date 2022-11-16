from ._csv_file_layouts import get_format, ColumnParseParameters
from pandas import DataFrame, read_csv, concat
from roguewave.timeseries_analysis.filtering import sos_filter
from numpy import cos, sin, pi
from typing import Iterator, List, Callable
from glob import glob
import os

_pattern = {
    "GPS": "????_GPS.csv",
    "FLT": "????_FLT.csv",
}


def apply_to_group(function: Callable[[DataFrame], DataFrame], dataframe: DataFrame):
    """
    Apply a function to each group seperately and recombine the result into a single dataframe

    :param function: Function to appply
    :param dataframe: Dataframe to apply function to
    :return:
    """
    groups = dataframe.groupby("group id")
    group_ids = [x[0] for x in groups]
    dataframes = [function(x[1]) for x in groups]
    return concat(dataframes, keys=group_ids)


def files_to_parse(path: str, pattern: str) -> Iterator[str]:
    files = glob(os.path.join(path, pattern))
    for file in sorted(files):
        yield file


def load_as_dataframe(
    files_to_parse: Iterator[str],
    csv_parsing_options: List[ColumnParseParameters],
) -> DataFrame:
    column_names = [x["column_name"] for x in csv_parsing_options]
    usecols = [x["column_name"] for x in csv_parsing_options if x["include"]]
    dtype = {x["column_name"]: x["dtype"] for x in csv_parsing_options}
    convert = [x["convert"] for x in csv_parsing_options if x["include"]]

    def process_file(file):
        df = read_csv(
            file,
            delimiter=",",
            header=0,
            names=column_names,
            dtype=dtype,
            on_bad_lines="skip",
            usecols=usecols,
        )
        df.dropna(inplace=True)

        for name, function in zip(usecols, convert):
            df[name] = function(df[name])

        return df

    source_files = list(files_to_parse)
    data_frames = [process_file(source_file) for source_file in source_files]
    return mark_continuous_groups(concat(data_frames, keys=source_files))


def mark_continuous_groups(df: DataFrame, sampling_interval=0.4):
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
    df["group id"] = (df["time"].diff() > sampling_interval + 0.01).cumsum()
    return df


def read_data(path, data_type) -> DataFrame:
    files = files_to_parse(path, _pattern[data_type])
    format = get_format(data_type)
    return load_as_dataframe(files, format)


def read_gps(path, postprocess=True) -> DataFrame:
    data = read_data(path, "GPS")

    if not postprocess:
        return data

    dataframe = DataFrame()
    dataframe["time"] = data["time"].values
    dataframe["latitude"] = (
        data["latitude degrees"].values + data["latitude minutes"].values
    )
    dataframe["longitude"] = (
        data["longitude degrees"].values + data["longitude minutes"].values
    )
    dataframe["z"] = data["z"].values

    angle = (90 - data["course over ground"].values / 1000) * pi / 180
    dataframe["u"] = cos(angle) * data["speed over ground"].values
    dataframe["v"] = sin(angle) * data["speed over ground"].values
    dataframe["w"] = data["w"].values
    dataframe["group id"] = data["group id"].values
    return dataframe


def read_displacement(path, postprocess=True) -> DataFrame:
    data = read_data(path, "FLT")
    if not postprocess:
        return data

    # Backward sos pass for phase correction
    def _process(_data: DataFrame):
        _data["x"] = sos_filter(_data["x"].values, "backward")
        _data["y"] = sos_filter(_data["y"].values, "backward")
        _data["z"] = sos_filter(_data["z"].values, "backward")
        return _data

    return apply_to_group(_process, data)
