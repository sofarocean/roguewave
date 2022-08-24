from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
import numpy
from pandas import DataFrame
from roguewave.tools.time import to_datetime_utc

SOFAR_STANDARD_NAMES_WAVE_BULK = {
    "latitude": "latitude",
    "longitude": "longitude",
    "timestamp": "time",
    "significant_waveheight": "significantWaveHeight",
    "peak_period": "peakPeriod",
    "mean_period": "meanPeriod",
    "peak_direction": "peakDirection",
    "peak_directional_spread": "peakDirectionalSpread",
    "mean_direction": "meanDirection",
    "mean_directional_spread": "meanDirectionalSpread",
    "peak_frequency": "peakFrequency"
}


@dataclass
class MetoceanData():
    latitude: float = numpy.nan
    longitude: float = numpy.nan
    time: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)

    _time: datetime = field(init=False, repr=False)

    @property
    def time(self) -> datetime:
        return self._time

    @time.setter
    def time(self, time):
        if isinstance(time, property):
            # Honestly not sure what is going on- but if we initialze
            # time with the default value of None a "property" object
            # gets passed instead of just "None". This catches that and
            # makes sure it all works. Bit of an edge case, should not
            # happen if initialized with value. This seems to be due to
            # the wave the dataclass and getters and setters interact and
            # the magic involved to make it work.
            time = time.fdel
        self._time = to_datetime_utc(time)

    def _convert(self, data,standard_sofar_names:bool=True ):
        if not standard_sofar_names:
            return data

        out = {}
        for key in data:
            if key in SOFAR_STANDARD_NAMES_WAVE_BULK:
                out[SOFAR_STANDARD_NAMES_WAVE_BULK[key]] = data[key]
            else:
                out[key] = data[key]
        return out

    def as_dict(self,standard_sofar_names:bool=True):
        return self._convert({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "time": self.time
            }, standard_sofar_names)

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

    def as_dict(self,standard_sofar_names:bool=True):
        return self._convert({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "time": self.time,
            "significant_waveheight": self.significant_waveheight,
            "peak_period": self.peak_period,
            "mean_period": self.mean_period,
            "peak_direction": self.peak_direction,
            "peak_directional_spread": self.peak_directional_spread,
            "mean_direction": self.mean_direction,
            "mean_directional_spread": self.mean_directional_spread,
            "peak_frequency": self.peak_frequency
        },standard_sofar_names)


@dataclass
class SSTData(MetoceanData):
    degrees: float = numpy.nan

    def as_dict(self,standard_sofar_names:bool=True):
        return self._convert({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "time": self.time,
            "degrees": self.degrees,
        },standard_sofar_names)

@dataclass
class WindData(MetoceanData):
    speed: float=numpy.nan
    direction: float = numpy.nan
    seasurfaceId: float = numpy.nan

    def as_dict(self,standard_sofar_names:bool=True):
        return self._convert({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.time,
            "speed": self.speed,
            "direction": self.direction,
            "seasurfaceId": self.seasurfaceId
        },standard_sofar_names)

@dataclass
class BarometricPressure(MetoceanData):
    units: str = ''
    value:float = numpy.nan
    unit_type: str = ''
    data_type_name: str = ''

    def as_dict(self,standard_sofar_names:bool=True):
        return self._convert({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.time,
            "units": self.units,
            "unit_type": self.unit_type,
            "data_type_name": self.data_type_name,
            "value": self.value
        }, standard_sofar_names)


def as_dataframe(bulk_wave_properties:List[MetoceanData],
                 standard_sofar_names=True):
    data = {}
    for bulk_propererties in bulk_wave_properties:
        dictionary = bulk_propererties.as_dict(standard_sofar_names)
        for key in dictionary:
            if key not in data:
                data[key] = []
            data[key].append( dictionary[key])

    for key in data:
        data[key] = data[key]

    df= DataFrame.from_dict(data)
    if standard_sofar_names:
        df.set_index('time', inplace=True)
    else:
        if 'timestamp' in df:
            df.set_index('timestamp', inplace=True)
        else:
            df.set_index('time', inplace=True)

    return df