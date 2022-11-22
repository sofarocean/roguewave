from numpy import float32, float64, int32, timedelta64
from typing import TypeVar, List, TypedDict, Callable
from ._spotter_constants import SpotterConstants, spotter_constants

_T = TypeVar("_T")


def do_nothing(x: _T) -> _T:
    return x


class ColumnParseParameters(TypedDict):
    header_name: str
    column_name: str
    dtype: type
    include: bool
    convert: Callable[[_T], _T]


_formats = {
    "GPS": [
        ColumnParseParameters(
            header_name="millis",
            column_name="milliseconds",
            dtype=float32,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="GPS_Epoch_Time(s)",
            column_name="time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="lat(deg)",
            column_name="latitude degrees",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="lat(min*1e5)",
            column_name="latitude minutes",
            dtype=float64,
            include=True,
            convert=lambda x: x / 1e5,
        ),
        ColumnParseParameters(
            header_name="lon(deg)",
            column_name="longitude degrees",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="long(min*1e5)",
            column_name="longitude minutes",
            dtype=float64,
            include=True,
            convert=lambda x: x / 1e5,
        ),
        ColumnParseParameters(
            header_name="el(m)",
            column_name="z",
            dtype=float32,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="SOG(mm/s)",
            column_name="speed over ground",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
        ColumnParseParameters(
            header_name="COG(deg*1000)",
            column_name="course over ground",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
        ColumnParseParameters(
            header_name="vert_vel(mm/s)",
            column_name="w",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
    ],
    "FLT": [
        ColumnParseParameters(
            header_name="millis",
            column_name="milliseconds",
            dtype=float32,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="GPS_Epoch_Time(s)",
            column_name="time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="outx(mm)",
            column_name="x",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
        ColumnParseParameters(
            header_name="outy(mm)",
            column_name="y",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
        ColumnParseParameters(
            header_name="outz(mm)",
            column_name="z",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1000,
        ),
        ColumnParseParameters(
            header_name="",
            column_name="init flag",
            dtype=str,
            include=False,
            convert=do_nothing,
        ),
    ],
    "LOC": [
        ColumnParseParameters(
            header_name="GPS_Epoch_Time(s)",
            column_name="time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="lat(deg)",
            column_name="latitude degrees",
            dtype=float32,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="at(min*1e5)",
            column_name="latitude minutes",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1e5,
        ),
        ColumnParseParameters(
            header_name="lon(deg)",
            column_name="longitude degrees",
            dtype=float32,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="long(min*1e5)",
            column_name="longitude minutes",
            dtype=float32,
            include=True,
            convert=lambda x: x / 1e5,
        ),
    ],
    "TIME": [
        ColumnParseParameters(
            header_name="GPS_Epoch_Time(s)",
            column_name="time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="lat(deg)",
            column_name="latitude degrees",
            dtype=float32,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="at(min*1e5)",
            column_name="latitude minutes",
            dtype=float32,
            include=False,
            convert=lambda x: x / 1e5,
        ),
        ColumnParseParameters(
            header_name="lon(deg)",
            column_name="longitude degrees",
            dtype=float32,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="long(min*1e5)",
            column_name="longitude minutes",
            dtype=float32,
            include=False,
            convert=lambda x: x / 1e5,
        ),
    ],
    "SPC": [
        ColumnParseParameters(
            header_name="type",
            column_name="type",
            dtype=str,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="millis",
            column_name="milliseconds",
            dtype=float32,
            include=False,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="t0_GPS_Epoch_Time(s)",
            column_name="time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="tN_GPS_Epoch_Time(s)",
            column_name="tN time",
            dtype=float64,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="ens_count",
            column_name="ensemble count",
            dtype=int32,
            include=True,
            convert=do_nothing,
        ),
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


def create_spectral_format(config: SpotterConstants) -> List[ColumnParseParameters]:
    if config is None:
        config = spotter_constants()

    _format = _formats["SPC"]
    sampling_interval = config["sampling_interval_gps"] / timedelta64(1, "s")

    sampling_frequency = 1 / (config["number_of_samples"] * sampling_interval)

    def _create(name) -> ColumnParseParameters:
        return ColumnParseParameters(
            header_name=name,
            column_name=name,
            dtype=float32,
            include=True,
            convert=lambda x: x / (1000000.0 * sampling_frequency),
        )

    column_names = spectral_column_names("flat", config)
    return _format + [_create(name) for name in column_names]


def get_format(
    file_type, config: SpotterConstants = None
) -> List[ColumnParseParameters]:
    if file_type == "SPC":
        return create_spectral_format(config)
    else:
        return _formats[file_type]
