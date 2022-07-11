"""
This file is part of pysofar: A client for interfacing with Sofar Oceans Spotter API

Contents: 2D Spectrum Class

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import numpy
import numpy.ma
from .wavespectrum import WaveSpectrum, WaveSpectrumInput
from roguewave.wavetheory.lineardispersion import inverse_intrinsic_dispersion_relation
from typing import List, Union
from roguewave.tools import datetime_to_iso_time_string
import scipy.ndimage
from numpy.ma import MaskedArray
import typing


class WaveSpectrum2DInput(WaveSpectrumInput):
    directions: Union[List[float], numpy.ndarray]

class WaveSpectrum2D(WaveSpectrum):
    def __init__(self,
                 wave_spectrum2D_input:WaveSpectrum2DInput
                 ):


        super().__init__(wave_spectrum2D_input)
        self.direction = numpy.array(wave_spectrum2D_input['directions'],
                                     dtype='float64')
        self._directional_difference = self._delta()

        self._frequency_peak_indices = None
        self._direction_peak_indices = None

    @property
    def e(self) -> numpy.array:
        if self._e is None:
            self._e = self._directional_moment('zero', 0, normalized=False)
        return self._e

    @property
    def a1(self) -> numpy.array:
        if self._a1 is None:
            self._a1 = self._directional_moment('a', 1, normalized=True)
        return self._a1

    @property
    def b1(self) -> numpy.array:
        if self._b1 is None:
            self._b1 = self._directional_moment('b', 1, normalized=True)
        return self._b1

    @property
    def a2(self) -> numpy.array:
        if self._a2 is None:
            self._a2 = self._directional_moment('a', 2, normalized=True)
        return self._a2

    @property
    def b2(self) -> numpy.array:
        if self._b2 is None:
            self._b2 = self._directional_moment('b', 2, normalized=True)
        return self._b2


    @WaveSpectrum.variance_density.setter
    def variance_density(self,val:numpy.ndarray):
        val = val.copy()
        mask = val < 0
        self._variance_density = MaskedArray(val,mask=mask,dtype='float64')
        self._update()

    def _update(self):
        super(WaveSpectrum2D, self)._update()
        self._a1 = None
        self._b1 = None
        self._a2 = None
        self._b2 = None
        self._e = None

    def _delta(self):
        angles = self.direction
        forward_diff = (numpy.diff(angles, append=angles[0]) + 180) % 360 - 180
        backward_diff = (numpy.diff(angles,
                                    prepend=angles[-1]) + 180) % 360 - 180
        return (forward_diff + backward_diff) / 2

    def _directional_moment(self, kind='zero', order=0,
                            normalized=True) -> numpy.array:
        delta = self._directional_difference
        if kind == 'a':
            harmonic = numpy.cos(self.radian_direction * order) * delta
        elif kind == 'b':
            harmonic = numpy.sin(self.radian_direction * order) * delta
        elif kind == 'zero':
            harmonic = delta
        else:
            raise Exception('Unknown moment')
        density = self.variance_density.filled(0)

        values = numpy.sum(density * harmonic[None, :], axis=-1)

        if normalized:
            scale = numpy.sum(density * delta[None, :], axis=-1)
            scale[scale == 0] = 1
        else:
            scale = 1
        return values / scale

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        range = self._range(fmin, fmax)
        return numpy.trapz(self.e[range] * self.frequency[range] ** power,
                           self.frequency[range])

    def _create_wave_spectrum_input(self)->WaveSpectrum2DInput:
        data = self.variance_density.filled(-1)
        return WaveSpectrum2DInput(
            frequency=list(self.frequency),
            directions=list(self.direction),
            varianceDensity=data.tolist(),
            timestamp=datetime_to_iso_time_string(self.timestamp),
            latitude=self.latitude,
            longitude=self.longitude,
        )

    def copy(self)->"WaveSpectrum2D":
        input = WaveSpectrum2DInput(
            frequency=self.frequency.copy(),
            directions=self.direction.copy(),
            varianceDensity=self.variance_density.copy(),
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude
        )
        return WaveSpectrum2D(input)

    def __add__(self, other:"WaveSpectrum2D")->"WaveSpectrum2D":
        spectrum = self.copy()

        self_dens = numpy.ma.MaskedArray(spectrum.variance_density)
        other_dens = numpy.ma.array(other.variance_density)

        mask = self_dens.mask & other_dens.mask

        spectrum.variance_density = numpy.ma.MaskedArray(self_dens.filled(0) +
            other_dens.filled(0), mask)
        return spectrum

    def __sub__(self, other: "WaveSpectrum2D") -> "WaveSpectrum2D":
        spectrum = self.copy()
        self_dens = numpy.ma.MaskedArray(spectrum.variance_density)
        other_dens = numpy.ma.array(other.variance_density)

        mask = self_dens.mask & other_dens.mask

        spectrum.variance_density = numpy.ma.MaskedArray(self_dens.filled(0) -
            other_dens.filled(0), mask)
        return spectrum

    def __neg__(self, other: "WaveSpectrum2D") -> "WaveSpectrum2D":
        spectrum = self.copy()
        spectrum.variance_density = -spectrum.variance_density
        return spectrum

    def extract(self,mask:numpy.ndarray)->"WaveSpectrum2D":
        spectrum = self.copy()
        density = spectrum.variance_density
        density[~mask] = 0
        density = numpy.ma.masked_array( density, mask=~mask)
        spectrum.variance_density = density
        return spectrum

    def peak_wavenumber_east(self, depth=numpy.inf):
        wave_number    = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi/180
        return wave_number * numpy.cos(wave_direction)

    def peak_wavenumber_north(self, depth=numpy.inf):
        wave_number    = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi/180
        return wave_number * numpy.cos(wave_direction)


def empty_spectrum2D_like(spectrum:WaveSpectrum2D)->WaveSpectrum2D:
    input = WaveSpectrum2DInput(
        frequency=spectrum.frequency.copy(),
        varianceDensity=numpy.zeros_like(spectrum.variance_density),
        timestamp=spectrum.timestamp,
        latitude=spectrum.latitude,
        longitude=spectrum.longitude,
        directions=spectrum.direction.copy()
    )
    return WaveSpectrum2D(input)




