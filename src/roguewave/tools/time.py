"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

from datetime import datetime, timezone
from xarray import DataArray
import numpy
from typing import Union, List, Tuple
from numpy import datetime64

scalar_input_types = Union[float, int, datetime, str, datetime64]
input_types = Union[scalar_input_types,List[scalar_input_types],numpy.ndarray]

def to_datetime_utc(time: input_types,to_scalar=False
                    ) -> Union[datetime, List[datetime]]:
    datetimes = _to_datetime_utc(time)
    if to_scalar and not isinstance(datetimes,datetime):
        return datetimes[0]
    else:
        return datetimes

def _to_datetime_utc(time: input_types) -> Union[datetime, List[datetime]]:
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

    elif isinstance(time,datetime64):
        time_stamp_seconds = numpy.datetime64(time,'s').astype('float64')
        time =  datetime.fromtimestamp( time_stamp_seconds ,tz=timezone.utc )
        return time

    elif isinstance(time,List) or isinstance(time,numpy.ndarray):
        return [ to_datetime_utc(x) for x in time ]

    elif isinstance(time,DataArray):
        time = time['time'].values
        return [ to_datetime_utc(x) for x in time ]

    else:
        return datetime.fromtimestamp(time, tz=timezone.utc)

def datetime_to_iso_time_string(time: input_types):
    if time is None:
        return None

    time = to_datetime_utc(time)
    return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def to_datetime64( time:Union[input_types,numpy.array,List,Tuple]
                   ) -> Union[numpy.ndarray,None]:
    """
    Convert time input to numpy ndarrays.
    :param time:
    :return:
    """
    if time is None:
        return None

    if isinstance(time, numpy.ndarray):
        return numpy.array([datetime64(to_datetime_utc(x),'s') for x in time])

    elif isinstance( time, List ) or isinstance( time, Tuple ):
        return numpy.array([datetime64(to_datetime_utc(x),'s') for x in time])

    elif isinstance( time, datetime64 ):
        return datetime64(time,'s')
    elif isinstance(time, datetime):
        return datetime64( int(time.timestamp()),'s')
    else:
        raise ValueError('unknown time type')
