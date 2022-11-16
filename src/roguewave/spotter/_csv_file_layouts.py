from numpy import float32, float64
from typing import TypeVar, List, TypedDict, Callable


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
            dtype=float32,
            include=True,
            convert=do_nothing,
        ),
        ColumnParseParameters(
            header_name="at(min*1e5)",
            column_name="latitude minutes",
            dtype=float32,
            include=True,
            convert=lambda x: x / 60e5,
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
            convert=lambda x: x / 60e5,
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
}


def get_format(file_type, **kwargs) -> List[ColumnParseParameters]:
    return _formats[file_type]
