"""
Contents: Abstract implementation Spectrum (see Spectrum1D and Spectrum2D for
implementations)

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import numpy
from roguewave.tools import to_datetime, datetime_to_iso_time_string
from typing import List, Tuple, Union
from .windSpotter import U10
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation, phase_velocity
from datetime import datetime
from numpy.ma import MaskedArray
import typing
from roguewave.metoceandata import WaveBulkData

class WaveSpectrum():
    """
    Base spectral class.
    """
    frequency_units = 'Hertz'
    angular_units = 'Degrees'
    spectral_density_units = 'm**2/Hertz'
    angular_convention = 'Wave travel direction (going-to), measured anti-clockwise from East'
    bulk_properties = (
        'm0', 'hm0', 'tm01', 'tm02', 'peak_period', 'peak_direction',
        'peak_directional_spread', 'mean_direction', 'mean_directional_spread',
        'peak_frequency',
        'peak_wavenumber', 'latitude', 'longitude', 'timestamp')

    def __init__(self,
                 frequency: Union[List[float],numpy.ndarray],
                 varianceDensity: Union[List[float],numpy.ndarray],
                 timestamp: Union[str, datetime, int, float],
                 latitude: Union[float, None],
                 longitude: Union[float, None]
                 ):
        self._a1 = None
        self._b1 = None
        self._a2 = None
        self._b2 = None
        self._e = None
        self.direction = None
        self._peak_index = None
        self._peak_wavenumber = None
        self.longitude = longitude
        self.latitude = latitude
        self.frequency = numpy.array(frequency)
        varianceDensity = numpy.array(varianceDensity)
        mask = (varianceDensity < 0) | (numpy.isnan(varianceDensity))
        self._variance_density = MaskedArray(varianceDensity, dtype='float64', mask=mask,fill_value=numpy.nan )
        self.timestamp = to_datetime(timestamp)


    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        pass

    def _create_wave_spectrum_input(self) -> dict:
        return {
            "frequency":self.frequency,
            "varianceDensity":self.variance_density,
            "timestamp":self.timestamp,
            "latitude":self.latitude,
            "longitude":self.longitude
        }

    @property
    def variance_density(self) -> numpy.ndarray:
        return self._variance_density

    def _update(self):
        self._peak_index = None
        self._peak_wavenumber = None

    @variance_density.setter
    def variance_density(self, val: numpy.ndarray):
        self._variance_density = MaskedArray(val)

    def _range(self, fmin=0.0, fmax=numpy.inf) -> numpy.ndarray:
        return (self.frequency >= fmin) & (self.frequency < fmax)

    @property
    def radian_direction(self) -> numpy.ndarray:
        return self.direction * numpy.pi / 180

    @property
    def radian_frequency(self) -> numpy.ndarray:
        return self.frequency * 2 * numpy.pi

    @property
    def e(self) -> numpy.array:
        return self._e

    @property
    def a1(self) -> numpy.array:
        return self._a1

    @property
    def b1(self) -> numpy.array:
        return self._b1

    @property
    def a2(self) -> numpy.array:
        return self._a2

    @property
    def b2(self) -> numpy.array:
        return self._b2

    @property
    def A1(self) -> numpy.array:
        return self.a1 * self.e

    @property
    def B1(self) -> numpy.array:
        return self.b1 * self.e

    @property
    def A2(self) -> numpy.array:
        return self.a2 * self.e

    @property
    def B2(self) -> numpy.array:
        return self.b2 * self.e

    def m0(self, fmin=0, fmax=numpy.inf) -> float:
        return self.frequency_moment(0, fmin, fmax)

    def m1(self, fmin=0, fmax=numpy.inf) -> float:
        return self.frequency_moment(1, fmin, fmax)

    def m2(self, fmin=0, fmax=numpy.inf) -> float:
        return self.frequency_moment(2, fmin, fmax)

    def hm0(self, fmin=0, fmax=numpy.inf) -> float:
        return 4 * numpy.sqrt(self.m0(fmin, fmax))

    def tm01(self, fmin=0, fmax=numpy.inf) -> float:
        return self.m0(fmin, fmax) / self.m1(fmin, fmax)

    def tm02(self, fmin=0, fmax=numpy.inf) -> float:
        return numpy.sqrt(self.m0(fmin, fmax) / self.m2(fmin, fmax))

    def peak_index(self, fmin=0, fmax=numpy.inf) -> float:
        if fmin == 0 and fmax == numpy.inf:
            if self._peak_index is None:
                self._peak_index = numpy.argmax(self.e)
            return self._peak_index

        range = self._range(fmin, fmax)
        tmp = self.e[:]
        tmp[~range] = 0
        return numpy.argmax(tmp)

    def peak_frequency(self, fmin=0, fmax=numpy.inf) -> float:
        return self.frequency[self.peak_index(fmin, fmax)]

    def peak_period(self, fmin=0, fmax=numpy.inf) -> float:
        return 1 / self.peak_frequency(fmin, fmax)

    def peak_direction(self, fmin=0, fmax=numpy.inf):
        index = self.peak_index(fmin, fmax)
        a1 = self.a1[index]
        b1 = self.b1[index]
        return self._mean_direction(a1, b1)

    def peak_directional_spread(self, fmin=0, fmax=numpy.inf):
        index = self.peak_index(fmin, fmax)
        a1 = self.a1[index]
        b1 = self.b1[index]
        return self._spread(a1, b1)

    @staticmethod
    def _mean_direction(a1, b1):
        return numpy.arctan2(b1, a1) * 180 / numpy.pi

    @staticmethod
    def _spread(a1, b1):
        return numpy.sqrt(
            2 - 2 * numpy.sqrt(a1 ** 2 + b1 ** 2)) * 180 / numpy.pi

    @property
    def mean_direction_per_frequency(self):
        return self._mean_direction(self.a1, self.b1)

    @property
    def mean_spread_per_frequency(self):
        return self._spread(self.a1, self.b1)

    def _spectral_weighted(self, property, fmin=0, fmax=numpy.inf):
        range = (self._range(fmin, fmax)) & numpy.isfinite(
            property) & numpy.isfinite(self.e)

        return numpy.trapz(property[range] * self.e[range],
                           self.frequency[range]) / self.m0(fmin, fmax)

    def mean_direction(self, fmin=0, fmax=numpy.inf):
        return self._mean_direction(self.mean_a1(fmin, fmax),
                                    self.mean_b1(fmin, fmax))

    def mean_directional_spread(self, fmin=0, fmax=numpy.inf):
        return self._spread(self.mean_a1(fmin, fmax), self.mean_b1(fmin, fmax))

    def mean_a1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a1, fmin, fmax)

    def mean_b1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.b1, fmin, fmax)

    def mean_a2(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a2, fmin, fmax)

    def mean_b2(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.b2, fmin, fmax)

    def U10(self, **kwargs) -> Tuple[float, float]:
        windspeed, winddirection, _ = U10(self.e, self.frequency, self.a1,
                                          self.b1, **kwargs)
        return windspeed, winddirection

    def Ustar(self, **kwargs) -> float:
        _, _, _Ustar = U10(self.e, self.frequency, self.a1,
                           self.b1, **kwargs)
        return _Ustar

    def wavenumber(self, depth=numpy.inf, **kwargs):
        return inverse_intrinsic_dispersion_relation(self.radian_frequency,
                                                     depth, **kwargs)

    def peak_wavenumber(self, depth=numpy.inf):

        if depth == numpy.inf:
            if self._peak_wavenumber is None:
                index = self.peak_index()
                self._peak_wavenumber = inverse_intrinsic_dispersion_relation(
                    self.radian_frequency[index], depth)
            return self._peak_wavenumber

        index = self.peak_index()
        return inverse_intrinsic_dispersion_relation(
            self.radian_frequency[index], depth)

    def peak_wavenumber_east(self, depth=numpy.inf):
        wave_number = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi / 180
        return wave_number * numpy.cos(wave_direction)

    def peak_wavenumber_north(self, depth=numpy.inf):
        wave_number = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi / 180
        return wave_number * numpy.cos(wave_direction)

    def peak_wave_age(self, ustar=None, depth=numpy.inf):
        if ustar is None:
            ustar = self.Ustar()
        return phase_velocity(self.peak_wavenumber(depth), depth) / ustar

    def wave_age(self, ustar=None, depth=numpy.inf):
        if ustar is None:
            ustar = self.Ustar()
        return phase_velocity(self.wavenumber(depth), depth) / ustar

    def bulk_variables(self) -> WaveBulkData:
        return WaveBulkData(
            significant_waveheight=self.significant_waveheight,
            mean_period=self.mean_period,
            peak_period=self.peak_period(),
            peak_direction=self.peak_direction(),
            peak_directional_spread=self.peak_directional_spread(),
            mean_direction=self.mean_direction(),
            mean_directional_spread=self.mean_directional_spread(),
            peak_frequency=self.peak_frequency(),
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude
        )

    def copy(self):
        pass

    def __add__(self, other) -> "WaveSpectrum":
        pass

    @property
    def significant_waveheight(self):
        return self.hm0()

    @property
    def mean_period(self):
        return self.tm01()


def extract_bulk_parameter(parameter, spectra: typing.List[WaveSpectrum]):
    if parameter == 'timestamp':
        output = numpy.empty(len(spectra), dtype='object')
    else:
        output = numpy.empty(len(spectra))

    for index, spectrum in enumerate(spectra):
        temp = getattr(spectrum, parameter)
        if callable(temp):
            output[index] = temp()
        else:
            output[index] = temp
    return output

