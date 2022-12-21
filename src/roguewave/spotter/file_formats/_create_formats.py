from json import dumps

_formats = {
    "GPS": {
        "sampling_interval_seconds": 0.4,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_GPS.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "millis",
                "dtype": "float64",
            },
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": "float64",
            },
            {
                "name": "lat(deg)",
                "dtype": "float64",
            },
            {
                "name": "lat(min*1e5)",
                "dtype": "float64",
            },
            {
                "name": "long(deg)",
                "dtype": "float64",
            },
            {
                "name": "long(min*1e5)",
                "dtype": "float64",
            },
            {
                "name": "el(m)",
                "dtype": "float64",
            },
            {
                "name": "SOG(mm_s)",
                "dtype": "float64",
            },
            {
                "name": "COG(deg*1000)",
                "dtype": "float64",
            },
            {
                "name": "vert_vel(mm_s)",
                "dtype": "float64",
            },
        ],
    },
    "FLT": {
        "sampling_interval_seconds": 0.4,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_FLT.csv",
        "ragged": False,
        "dropna": True,
        "drop": ["flag"],
        "columns": [
            {
                "name": "millis",
                "dtype": "float64",
            },
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": "float64",
            },
            {
                "name": "outx(mm)",
                "dtype": "float64",
            },
            {
                "name": "outy(mm)",
                "dtype": "float64",
            },
            {
                "name": "outz(mm)",
                "dtype": "float64",
            },
            {
                "name": "flag",
                "dtype": "str",
            },
        ],
    },
    "LOC": {
        "sampling_interval_seconds": 60,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_LOC.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": "float64",
            },
            {
                "name": "lat(deg)",
                "dtype": "float64",
            },
            {
                "name": "lat(min*1e5)",
                "dtype": "float64",
            },
            {
                "name": "long(deg)",
                "dtype": "float64",
            },
            {
                "name": "long(min*1e5)",
                "dtype": "float64",
            },
        ],
    },
    "TIME": {
        "sampling_interval_seconds": 60,
        "time_column": "GPS_Epoch_Time(s)",
        "pattern": "????_LOC.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "GPS_Epoch_Time(s)",
                "dtype": "float64",
            },
            {
                "name": "lat(deg)",
                "dtype": "float64",
            },
            {
                "name": "lat(min*1e5)",
                "dtype": "float64",
            },
            {
                "name": "long(deg)",
                "dtype": "float64",
            },
            {
                "name": "long(min*1e5)",
                "dtype": "float64",
            },
        ],
    },
    "GMN": {
        "sampling_interval_seconds": 10,
        "time_column": "millis",
        "pattern": "????_GMN.csv",
        "ragged": True,
        "dropna": False,
        "drop": [],
        "columns": [
            {
                "name": "millis",
                "dtype": "int64",
            },
            {
                "name": "noisePerMs",
                "dtype": "float64",
            },
            {
                "name": "agcCnt",
                "dtype": "float64",
            },
            {
                "name": "aStatus",
                "dtype": "float64",
            },
            {
                "name": "aPower",
                "dtype": "float64",
            },
            {
                "name": "jamInd",
                "dtype": "float64",
            },
            {
                "name": "jamStat",
                "dtype": "float64",
            },
            {
                "name": "ofsI",
                "dtype": "float64",
            },
            {
                "name": "magI",
                "dtype": "float64",
            },
            {
                "name": "ofsQ",
                "dtype": "float64",
            },
            {
                "name": "magQ",
                "dtype": "float64",
            },
            {
                "name": "VN(mm_s)",
                "dtype": "float64",
            },
            {
                "name": "VE(mm_s)",
                "dtype": "float64",
            },
            {
                "name": "VD(mm_s)",
                "dtype": "float64",
            },
            {
                "name": "avgSignalStrength",
                "dtype": "float64",
            },
            {
                "name": "SVs_used",
                "dtype": "float64",
            },
            {
                "name": "SVs_tracked",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_0",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_1",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_2",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_3",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_4",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_5",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_6",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_7",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_8",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_9",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_10",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_11",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_12",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_13",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_14",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_15",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_16",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_17",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_18",
                "dtype": "float64",
            },
            {
                "name": "signal_strengths_19",
                "dtype": "float64",
            },
        ],
    },
    "BARO_RAW": {
        "sampling_interval_seconds": 0.2,
        "time_column": "timestamp (ticks/UTC)",
        "pattern": "????_BARO_RAW.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "timestamp (ticks/UTC)",
                "dtype": "float64",
            },
            {
                "name": "temperature (C)",
                "dtype": "float64",
            },
            {
                "name": "pressure (mbar)",
                "dtype": "float64",
            },
        ],
    },
    "BARO": {
        "sampling_interval_seconds": 60,
        "time_column": "timestamp (ticks/UTC)",
        "pattern": "????_BARO.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "timestamp (ticks/UTC)",
                "dtype": "float64",
            },
            {
                "name": "pressure (mbar)",
                "dtype": "float64",
            },
        ],
    },
    "SST": {
        "sampling_interval_seconds": 60,
        "time_column": "timestamp (ticks/UTC)",
        "pattern": "????_SST.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "timestamp (ticks/UTC)",
                "dtype": "float64",
            },
            {
                "name": "temperature (C)",
                "dtype": "float64",
            },
        ],
    },
    "RAINDB": {
        "sampling_interval_seconds": 300,
        "time_column": "timestamp (ticks/UTC)",
        "pattern": "????_SST.csv",
        "ragged": False,
        "dropna": True,
        "drop": [],
        "columns": [
            {
                "name": "timestamp (ticks/UTC)",
                "dtype": "float64",
            },
            {
                "name": "level (dB)",
                "dtype": "float64",
            },
        ],
    },
}

if __name__ == "__main__":
    for k, v in _formats.items():
        with open(f"{k}.json", "wt") as file:
            str = dumps(v, indent=4)
            file.write(str)
