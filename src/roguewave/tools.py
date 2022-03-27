"""
This file is part of pysofar: A client for interfacing with Sofar Oceans Spotter API

Contents: Convinience functions to handle dates.

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

from datetime import datetime, timezone
import typing

def to_datetime(time: typing.Union[float, int, datetime, str]):
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
        except ValueError as e:
            return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")
    else:
        return datetime.fromtimestamp(time, tz=timezone.utc)

def datetime_to_iso_time_string(time: typing.Union[float, int, datetime, str]):
    if time is None:
        return None

    time = to_datetime(time)
    return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")