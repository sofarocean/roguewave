"""
Contents: Abstract implementation Spectrum (see Spectrum1D and Spectrum2D for
implementations)

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import numpy
from roguewave.tools import to_datetime, datetime_to_iso_time_string
from typing import TypedDict, List, Tuple, Union
from .windSpotter import U10
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation, phase_velocity
from datetime import datetime
from numpy.ma import MaskedArray
import typing


class WaveSpectrumInput(TypedDict):
    frequency: List[float]
    varianceDensity: List
    timestamp: Union[str, datetime, int, float]
    latitude: Union[float, None]
    longitude: Union[float, None]


class BulkVariables():
    def __init__(self, spectrum):
        if spectrum:
            self.m0 = spectrum.m0()
            self.hm0 = spectrum.hm0()
            self.tm01 = spectrum.tm01()
            self.tm02 = spectrum.tm02()
            self.peak_period = spectrum.peak_period()
            self.peak_direction = spectrum.peak_direction()
            self.peak_spread = spectrum.peak_spread()
            self.bulk_direction = spectrum.bulk_direction()
            self.bulk_spread = spectrum.bulk_spread()
            self.peak_frequency = spectrum.peak_frequency()
            self.peak_wavenumber = spectrum.peak_wavenumber()
            self.timestamp = spectrum.timestamp
            self.latitude = spectrum.latitude
            self.longitude = spectrum.longitude
        else:
            self._nanify()
            self.timestamp = numpy.nan
            self.latitude = numpy.nan
            self.longitude = numpy.nan

    def _nanify(self):
        self.m0 = numpy.nan
        self.hm0 = numpy.nan
        self.tm01 = numpy.nan
        self.tm02 = numpy.nan
        self.peak_period = numpy.nan
        self.peak_direction = numpy.nan
        self.peak_spread = numpy.nan
        self.bulk_direction = numpy.nan
        self.bulk_spread = numpy.nan
        self.peak_frequency = numpy.nan
        self.peak_wavenumber = numpy.nan


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
        'peak_spread','bulk_direction', 'bulk_spread', 'peak_frequency',
        'peak_wavenumber', 'latitude', 'longitude', 'timestamp')

    def __init__(self,
                 wave_spectrum_input: WaveSpectrumInput
                 ):
        self._a1 = None
        self._b1 = None
        self._a2 = None
        self._b2 = None
        self._e = None
        self.direction = None

        # Type conversions are needed because the JSON serializer does not accept float32
        self.frequency = numpy.array(wave_spectrum_input['frequency'],dtype='float64')
        self.variance_density = MaskedArray(
            wave_spectrum_input['varianceDensity'],dtype='float64')
        self.timestamp = to_datetime(wave_spectrum_input['timestamp'])
        self.longitude = float(wave_spectrum_input['longitude'])
        self.latitude = float(wave_spectrum_input['latitude'])

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        pass

    def _create_wave_spectrum_input(self) -> WaveSpectrumInput:
        return WaveSpectrumInput(
            frequency=list(self.frequency),
            varianceDensity=list(self.variance_density),
            timestamp=datetime_to_iso_time_string(self.timestamp),
            latitude=self.latitude,
            longitude=self.longitude
        )

    @property
    def variance_density(self) -> numpy.ndarray:
        return self._variance_density

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

    def peak_spread(self, fmin=0, fmax=numpy.inf):
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
    def mean_direction(self):
        return self._mean_direction(self.a1, self.b1)

    @property
    def mean_spread(self):
        return self._spread(self.a1, self.b1)

    def _spectral_weighted(self, property, fmin=0, fmax=numpy.inf):
        range = (self._range(fmin, fmax)) & numpy.isfinite(
            property) & numpy.isfinite(self.e)

        return numpy.trapz(property[range] * self.e[range],
                           self.frequency[range]) / self.m0(fmin, fmax)

    def bulk_direction(self, fmin=0, fmax=numpy.inf):
        return self._mean_direction(self.bulk_a1(fmin, fmax),
                                    self.bulk_b1(fmin, fmax))

    def bulk_spread(self, fmin=0, fmax=numpy.inf):
        return self._spread(self.bulk_a1(fmin, fmax), self.bulk_b1(fmin, fmax))

    def bulk_a1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a1, fmin, fmax)

    def bulk_b1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.b1, fmin, fmax)

    def bulk_a2(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a2, fmin, fmax)

    def bulk_b2(self, fmin=0, fmax=numpy.inf):
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

    def bulk_variables(self) -> BulkVariables:
        return BulkVariables(self)

    def copy(self):
        pass

    def __add__(self, other) -> "WaveSpectrum":
        pass


def extract_bulk_parameter(parameter, spectra: typing.List[WaveSpectrum]):

    if parameter == 'timestamp':
        output = numpy.empty(len(spectra),dtype='object')
    else:
        output = numpy.empty(len(spectra))

    for index, spectrum in enumerate(spectra):
        temp = getattr(spectrum, parameter)
        if callable(temp):
            output[index] = temp()
        else:
            output[index] = temp
    return output