from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy
from pandas import DataFrame

from roguewave.tools import to_datetime


@dataclass
class MetoceanData():
    latitude: float = numpy.nan
    longitude: float = numpy.nan
    timestamp: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)

    _timestamp: datetime = field(init=False, repr=False)

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, time):
        if isinstance(time, property):
            # Honestly not sure what is going on- but if we initialze
            # timestamp with the default value of None a "property" object
            # gets passed instead of just "None". This catches that and
            # makes sure it all works. Bit of an edge case, should not
            # happen if initialized with value. This seems to be due to
            # the wave the dataclass and getters and setters interact and
            # the magic involved to make it work.
            time = time.fdel
        self._timestamp = to_datetime(time)

    def as_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp
            }

@dataclass
class WaveBulkData(MetoceanData):

    significant_waveheight: float = numpy.nan
    peak_period: float = numpy.nan
    mean_period: float = numpy.nan
    peak_direction: float = numpy.nan
    peak_directional_spread: float = numpy.nan
    mean_direction: float = numpy.nan
    mean_directional_spread: float = numpy.nan
    peak_frequency: float = numpy.nan

    def as_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "significant_waveheight": self.significant_waveheight,
            "peak_period": self.peak_period,
            "mean_period": self.mean_period,
            "peak_direction": self.peak_direction,
            "peak_directional_spread": self.peak_directional_spread,
            "mean_direction": self.mean_direction,
            "mean_directional_spread": self.mean_directional_spread,
            "peak_frequency": self.peak_frequency
        }


@dataclass
class SSTData(MetoceanData):
    degrees: float = numpy.nan

    def as_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "degrees": self.degrees,
        }

@dataclass
class WindData(MetoceanData):
    speed: float=numpy.nan
    direction: float = numpy.nan
    seasurfaceId: float = numpy.nan

    def as_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "speed": self.speed,
            "direction": self.direction,
            "seasurfaceId": self.seasurfaceId
        }

@dataclass
class BarometricPressure(MetoceanData):
    units: str = ''
    value:float = numpy.nan
    unit_type: str = ''
    data_type_name: str = ''

    def as_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "units": self.units,
            "unit_type": self.unit_type,
            "data_type_name": self.data_type_name,
            "value": self.value
        }


def as_dataframe(bulk_wave_properties:List[MetoceanData]):
    data = {}
    for bulk_propererties in bulk_wave_properties:
        dictionary = bulk_propererties.as_dict()
        for key in dictionary:
            if key not in data:
                data[key] = []
            data[key].append( dictionary[key])

    for key in data:
        data[key] = numpy.array(data[key])

    df= DataFrame.from_dict(data)
    df.set_index('timestamp', inplace=True)
    return df