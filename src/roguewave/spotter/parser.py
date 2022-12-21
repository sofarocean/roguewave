import json
import os
import pandas
from typing import Literal, Callable
from datetime import datetime
from glob import glob
from typing import Iterator, Mapping
from numpy import timedelta64, zeros, float64, full
from numpy.typing import NDArray
from pandas import DataFrame, concat
from roguewave import to_datetime_utc
from roguewave.tools.time import to_datetime64
import xarray

CSV_TYPES = Literal[
    "FLT", "SPC", "GPS", "GMT", "LOC", "BARO", "BARO_RAW", "SST", "RAINDB"
]


# Main Functions
# ---------------------------------


def read_and_concatenate_spotter_csv(
    path,
    csv_type: CSV_TYPES,
    start_date: datetime = None,
    end_date: datetime = None,
    cache_as_netcdf=False,
) -> pandas.DataFrame:
    """
    Read data for a given data type from the given path.

    :param path: Path containing Spotter CSV files

    :param csv_type: One of the supported CSV file formats.

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
    if cache_as_netcdf:
        if os.path.exists(_netcdf_filename(path, csv_type)):
            dataset = xarray.open_dataset(_netcdf_filename(path, csv_type))
            return dataset.to_dataframe()

    csv_format = get_csv_file_format(csv_type)
    source_files = list(
        _files_to_parse(path, csv_format["pattern"], start_date, end_date)
    )

    if len(source_files) == 0:
        raise FileNotFoundError("No files to parse")

    data_frames = [
        _process_file(source_file, csv_format) for source_file in source_files
    ]

    # Fragmentation occurs for spectral data- to avoid performance issues we recreate the dataframe after the concat
    # here.
    dataframe = pandas.concat(data_frames).copy()
    dataframe.reset_index(inplace=True)
    dataframe = _mark_continuous_groups(
        dataframe, csv_format["sampling_interval_seconds"]
    )

    # Restrict to reequested start and end date (if applicable).
    if start_date is not None:
        start_date = to_datetime64(start_date)
        dataframe = dataframe[dataframe["time"] >= start_date]

    if end_date is not None:
        end_date = to_datetime64(end_date)
        dataframe = dataframe[dataframe["time"] < end_date]

    if cache_as_netcdf:
        save_as_netcdf(dataframe, path, csv_type)

    return dataframe


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

    time_file_format = get_csv_file_format("TIME")
    if (start_date is not None) or (end_date is not None):
        # If we are only interested in a range of data we do not want to parse all the text files. Specifically the
        # location, gps and spectral files can get very large (hundreds of mb) and consequenlty slow to read. Spotter
        # saves this data onboard split across multiple files (e.g. 0000_FLT.csv, 0001_FLT.csv etc) so we could just
        # load only those files containing the time range we need. We do not in general know which files contain which
        # time range though. Here we there try to read the timebase from the smallest data files (location files)
        # and use that to determine which files we read.

        # Read the files containing the timebase.
        time_files = glob(os.path.join(path, time_file_format["pattern"]))
        csv_parsing_options = get_csv_file_format("TIME")

        if not (len(time_files) == len(files)):
            # We need an equal number of files - otherwise something is amiss
            raise ValueError(
                "Number of location files needs to be the same as the number of files being read to use"
                "time filtering"
            )

        # For each data file and time base file do
        for file, time_file in zip(sorted(files), sorted(time_files)):

            # Read the time base file to get the time range in the XXXX_VAR.csv file
            time = pandas.read_csv(
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


def _process_file(file, csv_format, nrows=None):
    """
    Process a _single_ csv file of the form ????_TYPE.csv

    :param file: file name
    :param csv_format: the format of the csv file
    :param nrows: how many rows to read (default:None implies all rows)
    :return: dataframe containing date in the CSV file after quality control (filtering of nan rows etc.)
    """

    if not csv_format.get("ragged", False):
        df = pandas.read_csv(
            file,
            index_col=False,
            delimiter=",",
            skiprows=1,
            names=[x["name"] for x in csv_format["columns"]],
            on_bad_lines="skip",
            low_memory=False,
            nrows=nrows,
        )

    else:
        # Some spotter files are ragged (e.g. GMN) and may contain a variable number of columns. To handle
        # this we need to use the python parser.
        df = pandas.read_csv(
            file,
            index_col=False,
            delimiter=",",
            skiprows=1,
            names=[x["name"] for x in csv_format["columns"]],
            on_bad_lines="skip",
            engine="python",
            nrows=nrows,
        )

    # Assign the designated time column.
    df["time"] = df[csv_format["time_column"]]

    # If the time is in millis (milliseconds since system start) convert to unix epoch time in seconds (still float type
    # here)
    if csv_format["time_column"] == "millis":
        df["time"] = _milis_to_epoch(df["time"], file)

    # Convert the columns into the correct type. Coerce bad values to NaN values for numeric types.
    # (to note this is the reason we want to define numerics as floats, since integer do not contain a natural missing
    # value type)
    for column in csv_format["columns"]:
        name = column["name"]
        if column["dtype"] == "str":
            # Fill missing values in strings with blanks. Some Spotter files (FLT) contain a trailing string column
            # that does not always contain data and that gets filled with NA values. If we do not replace these with
            # blanks the subsequent dropping on NaN's in quality control would filter good data.
            df[name] = df[name].astype("string").fillna("")

        else:
            # TODO: what to do with malformed data in integer columns???
            df[name] = pandas.to_numeric(df[name], errors="coerce")

    # Drop NaN rows and remove any instances where we go back in time (only needed for raw GPS files).
    df = _quality_control(df, csv_format.get("dropna", True))

    # Convert time to proper pandas datetime format
    df["time"] = pandas.to_datetime(df["time"], unit="s")

    columns_to_drop = csv_format.get("drop", [])
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)

    return df


# Utility Functions
# ---------------------------------
def _netcdf_filename(path: str, csv_type: str) -> str:
    return os.path.join(path, f"{csv_type}.nc")


def save_as_netcdf(df: DataFrame, path, csv_type: CSV_TYPES) -> None:
    dataset = xarray.Dataset.from_dataframe(df)
    dataset.to_netcdf(_netcdf_filename(path, csv_type))


def _quality_control(dataframe: pandas.DataFrame, dropna) -> pandas.DataFrame:
    # Remove any rows with nan entries

    if dropna:
        dataframe = dataframe.dropna()

    # Here we check for negative intervals and only keep data with positive intervals. It needs to be recursively
    # applied since when removing an interval we may introduce a new negative interval. There are definitely more
    # efficient ways to implement this and if this ever is a bottleneck we should - until then we do it the ugly way :-)

    if "time" not in dataframe:
        return dataframe

    if dataframe.shape[0] < 1:
        return dataframe

    while True:
        positive_time = dataframe["time"].diff() > 0.0
        positive_time[0] = True
        if all(positive_time.values[1:]):
            break
        else:
            dataframe = dataframe.loc[positive_time]

    return dataframe


def _mark_continuous_groups(df: pandas.DataFrame, sampling_interval_seconds):
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

    if "time" in df:
        sampling_interval_seconds = timedelta64(
            int(sampling_interval_seconds * 1e9), "ns"
        )
        df["group id"] = (
            df["time"].diff() > sampling_interval_seconds + timedelta64(100, "ms")
        ).cumsum()
    else:
        df["group id"] = zeros(df.shape[0], dtype="int64")
    return df


def get_csv_file_format(csv_type: CSV_TYPES) -> Mapping:
    """
    Loads the CSV file format description from JSON files into a dict. All file formats
    are stored as JSON files in the ./file_formats directory.

    :param csv_type: which CSV format file to load.
    :return:
    """
    with open(
        os.path.join(os.path.dirname(__file__), f"file_formats/{csv_type}.json")
    ) as file:
        format = json.loads(file.read())
    return format


def _milis_to_epoch(millis: NDArray[float64], millis_file: str) -> NDArray[float64]:
    """
    Convert Spotter "millis" into unix epoch time stamp. Millis measure time elapsed in miliseconds since the start of
    the system. To translate these into a timestamp referenced to the Unix epoch we use the FLT files- since these
    contain both a Millis and Epoch time timestamp allowing us to determine the delta.

    :param millis: array of millis
    :param millis_file: filename of the array that contained the millis (used to get the proper FLT file)
    :return: array of corresponding epoch times
    """

    #
    if len(millis) == 0:
        # If len 0 there is nothing to convert
        return millis

    # get the path and filename of the milis file, and use it to deduce the corresponding
    # _FLT filename
    path, filename = os.path.split(millis_file)
    flt_file = os.path.join(path, filename.split("_")[0] + "_FLT.csv")

    # Get the data format for the flt file, and load 1 line of data from it
    csv_format = get_csv_file_format("FLT")
    data = _process_file(flt_file, csv_format, 1)

    # Use the millis and epoch time to deduce the systems epoch (aka startup) time.
    epoch = data[csv_format["time_column"]].values[0] - data["millis"].values[0] / 1000

    # Add the epoch to millis converted to seconds and return result.
    return epoch + millis / 1000


def apply_to_group(function: Callable[[DataFrame], DataFrame], dataframe: DataFrame):
    """
    Apply a function to each group seperately and recombine the result into a single dataframe.

    :param function: Function to appply
    :param dataframe: Dataframe to apply function to
    :return:
    """
    groups = dataframe.groupby("group id")

    dataframes = []
    for group_id, df in groups:
        _df = function(df)
        if "group id" not in _df:
            _df["group id"] = full(_df.shape[0], group_id)
        dataframes.append(_df)

    return concat(dataframes)
