from pandas import read_csv, concat, DataFrame, Timestamp
from roguewave import to_datetime_utc
from numpy import pi, cos, sin, int64, int32, float32, float64
from typing import Iterator, TypeVar
from glob import glob
import os

_T = TypeVar("_T")


def do_nothing(x: _T) -> _T:
    return x


GPS_HEADER = {
    "millis": {"dtype": float32, "include": False, "convert": do_nothing},
    "GPS_Epoch_Time(s)": {"dtype": float64, "include": True, "convert": do_nothing},
    "lat(deg)": {"dtype": float32, "include": True, "convert": do_nothing},
    "lat(min*1e5)": {"dtype": float32, "include": True, "convert": lambda x: x / 60e5},
    "long(deg)": {"dtype": float32, "include": True, "convert": do_nothing},
    "long(min*1e5)": {"dtype": float32, "include": True, "convert": lambda x: x / 60e5},
    "el(m)": {"dtype": float32, "include": True, "convert": do_nothing},
    "SOG(mm/s)": {"dtype": float32, "include": True, "convert": lambda x: x / 1000},
    "COG(deg*1000)": {"dtype": float32, "include": True, "convert": lambda x: x / 1000},
    "vert_vel(mm/s)": {
        "dtype": float32,
        "include": True,
        "convert": lambda x: x / 1000,
    },
}

DISP_CSV_PARSING = {
    "millis": {"dtype": float32, "include": False, "convert": do_nothing},
    "time": {"dtype": float64, "include": True, "convert": do_nothing},
    "x": {"dtype": float32, "include": True, "convert": lambda x: x / 1000},
    "y": {"dtype": float32, "include": True, "convert": lambda x: x / 1000},
    "z": {"dtype": float32, "include": True, "convert": lambda x: x / 1000},
    "init": {"dtype": str, "include": False, "convert": do_nothing},
}


def files_to_parse(path: str, pattern: str) -> Iterator[str]:
    files = glob(os.path.join(path, pattern))
    for file in sorted(files):
        yield file


def load_as_dataframe(
    files_to_parse: Iterator[str],
    csv_parsing_options: dict,
    postprocess_dataframe=lambda x: x,
) -> DataFrame:
    names = list(csv_parsing_options.keys())
    columns = [key for key in names if csv_parsing_options[key]["include"]]
    dtype = {key: csv_parsing_options[key]["dtype"] for key in names}

    def process_file(file):
        df = read_csv(
            file,
            delimiter=",",
            header=0,
            names=names,
            dtype=dtype,
            on_bad_lines="skip",
            usecols=columns,
        )
        df.dropna(inplace=True)

        for name in columns:
            df[name] = csv_parsing_options[name]["convert"](df[name])

        return postprocess_dataframe(df)

    source_files = list(files_to_parse)
    data_frames = [process_file(source_file) for source_file in source_files]
    return mark_continuous_groups(concat(data_frames, keys=source_files))


def mark_continuous_groups(df: DataFrame, sampling_interval=0.4):
    df["continuous_group_marker"] = (
        df["time"].diff() > sampling_interval + 0.01
    ).cumsum()
    return df


def load_gps(path: str) -> DataFrame:
    def process(raw_dataframe: DataFrame) -> DataFrame:
        dataframe = DataFrame()
        dataframe["time"] = raw_dataframe["GPS_Epoch_Time(s)"].values
        dataframe["latitude"] = (
            raw_dataframe["lat(deg)"].values
            + raw_dataframe["lat(min*1e5)"].values / 60e5
        )
        dataframe["longitude"] = (
            raw_dataframe["long(deg)"].values
            + raw_dataframe["long(min*1e5)"].values / 60e5
        )
        dataframe["z"] = raw_dataframe["el(m)"].values

        angle = (90 - raw_dataframe["COG(deg*1000)"].values / 1000) * pi / 180
        dataframe["u"] = cos(angle) * raw_dataframe["SOG(mm/s)"].values
        dataframe["v"] = sin(angle) * raw_dataframe["SOG(mm/s)"].values
        dataframe["w"] = raw_dataframe["vert_vel(mm/s)"].values
        return dataframe

    return load_as_dataframe(
        files_to_parse(path, "????_GPS.csv"),
        csv_parsing_options=GPS_HEADER,
        postprocess_dataframe=process,
    )


def load_displacement(path):
    # millis, GPS_Epoch_Time(s), outx(mm), outy(mm), outz(mm)
    return load_as_dataframe(
        files_to_parse(path, "????_FLT.csv"),
        csv_parsing_options=DISP_CSV_PARSING,
        postprocess_dataframe=do_nothing,
    )


if __name__ == "__main__":
    path = "/Users/pietersmit/Downloads/Sunflower13/log"
    df = load_gps(path)
    print(df.shape)
