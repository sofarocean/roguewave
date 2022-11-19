"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit


"""
from datetime import datetime, timezone, timedelta
from xarray import DataArray
from typing import Union, Sequence
from numpy import datetime64, ndarray, array, timedelta64
from numpy.typing import NDArray
from numbers import Number
from pandas import Series

scalar_input_types = Union[str, float, int, datetime, datetime64]
input_types = Union[scalar_input_types, Sequence[scalar_input_types]]


def to_datetime_utc(time: input_types) -> Union[datetime, Sequence[datetime], None]:
    """
    Output datetimes are garantueed to be in the UTC timezone. For timezone naive input the timezone is assumed to be
    UTC. None as input is translated to None as output to allow for cases where time is optional. Note that the
    implementation works with heterogeneous sequences.

    :param time: Time, is either a valid scalar time type or a sequence of time types.
    :return: If the input is a sequence, the output is a sequence of datetimes, otherwise it is a scalar datetime.

    """

    if time is None:
        return None

    if isinstance(time, (DataArray, list, tuple, ndarray, Series)):
        # if this is a sequence type, recursively call to datetime on the sequence
        if isinstance(time, (DataArray, Series)):
            time = time.values

        return [to_datetime_utc(x) for x in time]

    else:
        # if this is a scalar type, do the appropriate conversion.
        if isinstance(time, datetime):
            if time.tzinfo is None:
                time = time.replace(tzinfo=timezone.utc)
            return time.astimezone(timezone.utc)

        elif isinstance(time, str):
            if time[-1] == "Z":
                # From isoformat does not parse "Z" as a valid timezone designator. This should be fixed in Python 11.
                time = time[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(time)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return dt.astimezone(
                    timezone.utc
                )  # datetime.fromisoformat(time).astimezone(timezone.utc)
            except ValueError:
                return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(
                    timezone.utc
                )

        elif isinstance(time, datetime64):
            # We first cast datetime64 explicitly to seconds and then as a float to allow for fractional seconds.
            return datetime.fromtimestamp(
                datetime64(time, "s").astype("float64"), tz=timezone.utc
            ).replace(tzinfo=timezone.utc)
        elif isinstance(time, Number):
            return datetime.fromtimestamp(time, tz=timezone.utc)

        else:
            raise ValueError(f"Unknown time type: {type(time)} with value {time}")


def datetime_to_iso_time_string(time: scalar_input_types):
    if time is None:
        return None

    time = to_datetime_utc(time)
    return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def datetime64_to_timestamp(time):
    return (time - datetime64(0, "s")) / timedelta64(1, "s")


def to_datetime64(time) -> Union[None, datetime64, NDArray[datetime64]]:
    """
    Convert time input to numpy ndarrays.
    :param time:
    :return:
    """
    if time is None:
        return None

    # Convert to a (sequence of) UTC datetime(s)
    time = to_datetime_utc(time)

    if isinstance(time, datetime):
        # If a datetime- do conversion
        return datetime64(int(time.timestamp()), "s")
    else:
        # if  a sequence, do list comprehension and return an array
        return array([datetime64(int(x.timestamp()), "s") for x in time])


def time_from_timeint(t: int) -> timedelta:
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


def date_from_dateint(t: int) -> datetime:
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
    date_int: int, time_int: int, as_datetime64=False
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
