"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

from datetime import datetime, timezone, timedelta
from xarray import DataArray
from typing import Union, List, Sequence
from numpy.typing import NDArray
from numpy import datetime64, ndarray, array

scalar_input_types = Union[float, int, datetime, str, datetime64]
input_types = Union[scalar_input_types, List[scalar_input_types], NDArray]


def to_datetime_utc(time: input_types) -> Union[datetime, Sequence[datetime], None]:

    if time is None:
        return None

    if isinstance(time, datetime):
        return time.astimezone(timezone.utc)

    elif isinstance(time, str):
        if time[-1] == "Z":
            # From isoformat does not parse "Z" as a valid timezone designator
            time = time[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(time)
        except ValueError:
            return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")

    elif isinstance(time, datetime64):
        return datetime.fromtimestamp(
            datetime64(time, "s").astype("float64"), tz=timezone.utc
        )

    elif isinstance(time, List) or isinstance(time, ndarray):
        return [to_datetime_utc(x) for x in time]

    elif isinstance(time, DataArray):
        return to_datetime_utc(time.values)

    else:
        return datetime.fromtimestamp(time, tz=timezone.utc)


def datetime_to_iso_time_string(time: scalar_input_types):
    if time is None:
        return None

    time = to_datetime_utc(time)
    return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def to_datetime64(time) -> Union[None, datetime64, Sequence[datetime64]]:
    """
    Convert time input to numpy ndarrays.
    :param time:
    :return:
    """
    if time is None:
        return None

    time = to_datetime_utc(time)
    if isinstance(time, datetime):
        return datetime64(int(time.timestamp()), "s")
    else:
        return array([datetime64(int(x.timestamp()), "s") for x in time])


def time_from_timeint(t) -> timedelta:
    """
    unpack a timedelta from a time given as an integer in the form "hhmmss" e.g. 201813 for 20:18:13
    """
    if t >= 10000:
        hours = t // 10000
        minutes = (t - hours * 10000) // 100
        seconds = t - hours * 10000 - minutes * 100
    elif t >= 100:
        hours = t // 100
        minutes = t - hours * 100
        seconds = 0
    else:
        hours = t
        minutes = 0
        seconds = 0

    return timedelta(seconds=(hours * 3600 + minutes * 60 + seconds))


def date_from_dateint(t) -> datetime:
    """
    unpack a datetime from a date given as an integer in the form "yyyymmdd" or "yymmdd" e.g. 20221109 for 2022-11-09
    or 221109 for 2022-11-09
    """

    if t > 1000000:
        years = t // 10000
        months = (t - years * 10000) // 100
        days = t - years * 10000 - months * 100
    else:
        years = t // 10000
        months = (t - years * 10000) // 100
        days = t - years * 10000 - months * 100
        years = years + 2000

    return datetime(years, months, days, tzinfo=timezone.utc)


def datetime_from_time_and_date_integers(
    date_int, time_int, as_datetime64=False
) -> Union[datetime, datetime64]:
    """
    Convert a date and time given as integed encoded in the form "yyyymmdd" and "hhmm" _or_ "hhmmss" to a datetime
    :param date_int: integer of the form yyyymmdd
    :param time_int: time of the form "hhmm" or "hhmmss"
    :return:
    """
    dt = date_from_dateint(date_int) + time_from_timeint(time_int)
    if as_datetime64:
        return to_datetime64(dt)
    else:
        return dt
