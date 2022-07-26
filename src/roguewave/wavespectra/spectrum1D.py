"""
Contents: 1D Spectrum Class

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

import numpy
from .wavespectrum import WaveSpectrum
from typing import List, Union
from datetime import datetime



class WaveSpectrum1D(WaveSpectrum):
    spectral_density_units = 'm**2/Hertz'

    def __init__(self,
                 frequency: Union[List[float], numpy.ndarray],
                 varianceDensity: Union[List[float], numpy.ndarray],
                 a1: Union[List[float], numpy.ndarray],
                 b1: Union[List[float], numpy.ndarray],
                 a2: Union[List[float], numpy.ndarray],
                 b2: Union[List[float], numpy.ndarray],
                 timestamp: Union[str, datetime, int, float],
                 latitude: Union[float, None],
                 longitude: Union[float, None],
                 **kwargs
                 ):
        super().__init__(frequency=frequency, varianceDensity=varianceDensity,
                         timestamp=timestamp,latitude=latitude,longitude=longitude)
        self._a1 = numpy.array(a1)
        self._b1 = numpy.array(b1)
        self._b2 = numpy.array(b2)
        self._a2 = numpy.array(a2)
        self._e = numpy.array(self.variance_density)

    @WaveSpectrum.variance_density.setter
    def variance_density(self,val:numpy.ndarray):
        self._variance_density = val
        self._e = val

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        range = (self._range(fmin, fmax)) & numpy.isfinite(self.e)

        return numpy.trapz(
            self.variance_density[range] * self.frequency[range] ** power,
            self.frequency[range])

    def _create_wave_spectrum_input(self) -> dict:
        return {
            "frequency": self.frequency,
            "varianceDensity": self.variance_density,
            "a1": self.a1,
            "b1": self.b1,
            "a2": self.a2,
            "b2": self.b2,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    def copy(self) -> "WaveSpectrum1D":
        return WaveSpectrum1D(
            frequency=self.frequency.copy(),
            varianceDensity=self.variance_density.copy(),
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude,
            a1=self.a1,
            b1=self.b1,
            a2=self.a2,
            b2=self.b2)

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
    return WaveSpectrum1D(
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


