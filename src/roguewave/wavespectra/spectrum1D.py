"""
Contents: 1D Spectrum Class

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

import numpy
from .wavespectrum import WaveSpectrum, WaveSpectrumInput
from roguewave.tools import datetime_to_iso_time_string
from typing import List, Union
from datetime import timedelta


class WaveSpectrum1DInput(WaveSpectrumInput):
    a1: Union[List[float], numpy.ndarray]
    b1: Union[List[float], numpy.ndarray]
    a2: Union[List[float], numpy.ndarray]
    b2: Union[List[float], numpy.ndarray]


class WaveSpectrum1D(WaveSpectrum):
    spectral_density_units = 'm**2/Hertz'

    def __init__(self,
                 wave_spectrum1D_input: WaveSpectrum1DInput
                 ):
        super().__init__(wave_spectrum1D_input)
        self._a1 = numpy.array(wave_spectrum1D_input['a1'])
        self._b1 = numpy.array(wave_spectrum1D_input['b1'])
        self._b2 = numpy.array(wave_spectrum1D_input['b2'])
        self._a2 = numpy.array(wave_spectrum1D_input['a2'])
        self._e = numpy.array(wave_spectrum1D_input['varianceDensity'])

    @WaveSpectrum.variance_density.setter
    def variance_density(self,val:numpy.ndarray):
        self._variance_density = val
        self._e = val

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        range = (self._range(fmin, fmax)) & numpy.isfinite(self.e)

        return numpy.trapz(
            self.variance_density[range] * self.frequency[range] ** power,
            self.frequency[range])

    def _create_wave_spectrum_input(self) -> WaveSpectrum1DInput:
        return WaveSpectrum1DInput(
            frequency=list(self.frequency),
            varianceDensity=list(self.variance_density),
            timestamp=datetime_to_iso_time_string(self.timestamp),
            latitude=self.latitude,
            longitude=self.longitude,
            a1=list(self.a1),
            b1=list(self.b1),
            a2=list(self.a2),
            b2=list(self.b2)
        )

    def copy(self) -> "WaveSpectrum1D":
        input = WaveSpectrum1DInput(
            frequency=self.frequency.copy(),
            varianceDensity=self.variance_density.copy(),
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude,
            a1=list(self.a1),
            b1=list(self.b1),
            a2=list(self.a2),
            b2=list(self.b2)
        )
        return WaveSpectrum1D(input)

    def __add__(self, other:"WaveSpectrum1D")->"WaveSpectrum1D":
        spectrum = self.copy()
        spectrum.variance_density = (spectrum.variance_density +
            other.variance_density)
        return spectrum

    def __sub__(self, other:"WaveSpectrum1D")->"WaveSpectrum1D":
        spectrum = self.copy()
        spectrum.variance_density = (spectrum.variance_density -
            other.variance_density)
        return spectrum

    def __neg__(self, other:"WaveSpectrum1D")->"WaveSpectrum1D":
        spectrum = self.copy()
        spectrum.variance_density = -spectrum.variance_density
        return spectrum


def empty_spectrum1D_like(spectrum: WaveSpectrum1D) -> WaveSpectrum1D:
    input = WaveSpectrum1DInput(
        frequency=spectrum.frequency.copy(),
        varianceDensity=numpy.zeros_like(spectrum.variance_density),
        timestamp=spectrum.timestamp,
        latitude=spectrum.latitude,
        longitude=spectrum.longitude,
        a1=list(spectrum.a1),
        b1=list(spectrum.b1),
        a2=list(spectrum.a2),
        b2=list(spectrum.b2)
    )
    return WaveSpectrum1D(input)


