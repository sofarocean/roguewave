from json import dumps

_format = {
    "sampling_interval_seconds": 3600,
    "sampling_interval_gps": 0.4,
    "nfft": 256,
    "time_column": "t0_GPS_Epoch_Time(s)",
    "pattern": "????_SPC.csv",
    "columns": [
        {
            "name": "type",
            "dtype": "str",
        },
        {
            "name": "millis",
            "dtype": "float64",
        },
        {
            "name": "t0_GPS_Epoch_Time(s)",
            "dtype": "float64",
        },
        {
            "name": "tN_GPS_Epoch_Time(s)",
            "dtype": "float64",
        },
        {
            "name": "ens_count",
            "dtype": "int64",
        },
    ],
}

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


def spectral_column_names(groupedby="frequency"):
    names = _spectral_column_names["default"]

    columns = []
    for index in range(0, _format["nfft"] // 2):
        columns += [name + f"_{index}" for name in names]
    return columns


def create_spectral_format():
    def _create(name):
        return {
            "name": name,
            "dtype": "float64",
        }

    column_names = spectral_column_names("flat")
    _format["columns"] += [_create(name) for name in column_names]
    return _format


if __name__ == "__main__":
    format = create_spectral_format()
    with open("SPC.json", "wt") as file:
        str = dumps(format, indent=4)
        file.write(str)
