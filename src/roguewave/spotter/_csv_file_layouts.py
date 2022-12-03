from numpy import float32, float64, int32, timedelta64
from typing import TypeVar, List, TypedDict, Callable
from ._spotter_constants import SpotterConstants, spotter_constants

_T = TypeVar("_T")

file_name_pattern = {
    "GPS": "????_GPS.csv",
    "FLT": "????_FLT.csv",
    "LOC": "????_LOC.csv",
    "SPC": "????_SPC.csv",
    "TIME": "????_LOC.csv",
}

_formats = {
    "GPS":
        {
            "sampling_interval_seconds": 0.4,
            "time_column": "GPS_Epoch_Time(s)",
            "pattern": "????_GPS.csv",
            "columns": [
                {
                    "name": "millis",
                    "dtype": 'float64',
                },
                {
                    "name": "GPS_Epoch_Time(s)",
                    "dtype": 'float64',
                },
                {
                    "name": "lat(deg)",
                    "dtype": 'float64',
                },
                {
                    "name": "lat(min*1e5)",
                    "dtype": 'float64',
                },
                {
                    "name": "long(deg)",
                    "dtype": 'float64',
                },
                {
                    "name": "long(min*1e5)",
                    "dtype": 'float64',
                },
                {
                    "name": "el(m)",
                    "dtype": 'float64',
                },
                {
                    "name": "SOG(mm/s)",
                    "dtype": 'float64',
                },
                {
                    "name": "COG(deg*1000)",
                    "dtype": 'float64',
                },
                {
                    "name": "vert_vel(mm/s)",
                    "dtype": 'float64',
                },
            ],
        },
    "FLT": {
        "sampling_interval_seconds": 0.4,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_FLT.csv",
        "columns": [
            {
                "name": "millis",
                "dtype": 'float64',
            },
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": 'float64',
            },
            {
                "name": "outx(mm)",
                "dtype": 'float64',
            },
            {
                "name": "outy(mm)",
                "dtype": 'float64',
            },
            {
                "name": "outz(mm)",
                "dtype": 'float64',
            },
            {
                "name": "flag",
                "dtype": 'str',
            },
        ],
    },
    "LOC": {
        "sampling_interval_seconds": 60,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_LOC.csv",
        "columns": [
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": 'float64',
            },
            {
                "name": "lat(deg)",
                "dtype": 'float64',
            },
            {
                "name": "lat(min*1e5)",
                "dtype": 'float64',
            },
            {
                "name": "long(deg)",
                "dtype": 'float64',
            },
            {
                "name": "long(min*1e5)",
                "dtype": 'float64',
            },
        ],
    },
    "TIME": {
        "sampling_interval_seconds": 60,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_LOC.csv",
        "columns": [
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": 'float64',
            },
            {
                "name": "lat(deg)",
                "dtype": 'float64',
            },
            {
                "name": "lat(min*1e5)",
                "dtype": 'float64',
            },
            {
                "name": "long(deg)",
                "dtype": 'float64',
            },
            {
                "name": "long(min*1e5)",
                "dtype": 'float64',
            },
        ],
    },
    "SPC": {
        "sampling_interval_seconds": 3600,
        "time_column": "t0_GPS_Epoch_Time(s)",
        "pattern": "????_SPC.csv",
        "columns": [
            {
                "name": "type",
                "dtype": 'str',
            },
            {
                "name": "millis",
                "dtype": 'float64',
            },
            {
                "name": "t0_GPS_Epoch_Time(s)",
                "dtype": 'float64',
            },
            {
                "name": "tN_GPS_Epoch_Time(s)",
                "dtype": 'float64',
            },
            {
                "name": "ens_count",
                "dtype": "int64",
            },
        ],
    },
    "GMN": {
        "sampling_interval_seconds": 0.4,
        "time_column": "millis",
        "pattern": "????_GPS.csv",
        "columns": [
            {
                "name": "millis",
                "dtype": 'float64',
            },
            {
                "name": "noisePerMs",
                "dtype": 'float64',
            },
            {
                "name": "agcCnt",
                "dtype": 'float64',
            },
        ],
    }
}
# millis,noisePerMs,agcCnt,aStatus,aPower,jamInd,jamStat,ofsI,magI,ofsQ,magQ,VN(mm/s),VE(mm/s),VD(mm/s),avgSignalStrength,SVs_used,SVs_tracked,signal_strengths


_spectral_column_names = {
    "default": [
        "Sxx_re",
        "Syy_re",
        "Szz_re",
        "Sxy_re",
        "Szx_re",
        "Szy_re",
        "Sxx_im",
        "Syy_im",
        "Szz_im",
        "Sxy_im",
        "Szx_im",
        "Szy_im",
    ]
}


def spectral_column_names(groupedby="frequency", config: SpotterConstants = None):
    if config is None:
        config = spotter_constants()

    if config["n_channel"] is False:
        names = _spectral_column_names["default"]
    else:
        raise ValueError("not implemented yet")

    columns = []
    if groupedby == "frequency":
        for index in range(0, config["number_of_frequencies"]):
            columns.append((index, [name + f"_{index}" for name in names]))
        return columns
    if groupedby == "flat":
        for index in range(0, config["number_of_frequencies"]):
            columns += [name + f"_{index}" for name in names]
        return columns
    elif groupedby == "kind":
        for name in names:
            columns.append(
                (
                    name,
                    [
                        name + f"_{index}"
                        for index in range(0, config["number_of_frequencies"])
                    ],
                )
            )
        return columns
    else:
        raise ValueError("Unknown grouping")


def create_spectral_format(config: SpotterConstants):
    if config is None:
        config = spotter_constants()

    _format = _formats["SPC"].copy()
    def _create(name):
        return {
            "name": name,
            "dtype": 'float64',
        }

    column_names = spectral_column_names("flat", config)
    _format['columns'] += [_create(name) for name in column_names]
    return _format


def get_format(
        file_type, config: SpotterConstants = None
):
    if file_type == "SPC":
        return create_spectral_format(config)
    else:
        return _formats[file_type]
