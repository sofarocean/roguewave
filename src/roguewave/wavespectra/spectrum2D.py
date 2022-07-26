"""
This file is part of pysofar: A client for interfacing with Sofar Oceans Spotter API

Contents: 2D Spectrum Class

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import numpy
import numpy.ma
from .wavespectrum import WaveSpectrum
from typing import List, Union
from numpy.ma import MaskedArray
from numba import njit
from functools import cached_property
from datetime import datetime

class WaveSpectrum2D(WaveSpectrum):
    def __init__(self,
                 frequency: Union[List[float],numpy.ndarray],
                 varianceDensity: Union[List[float],numpy.ndarray],
                 directions: Union[List[float], numpy.ndarray],
                 timestamp: Union[str, datetime, int, float],
                 latitude: Union[float, None],
                 longitude: Union[float, None],
                 **kwargs
                 ):
        super().__init__(frequency=frequency, varianceDensity=varianceDensity,
                         timestamp=timestamp,latitude=latitude,longitude=longitude)
        self.direction = numpy.array(directions)
        self._directional_difference = self._delta()

        self._frequency_peak_indices = None
        self._direction_peak_indices = None

    @cached_property
    def e(self) -> numpy.array:
        return _directional_moment( self.variance_density.filled(0),
                                     self.radian_direction,
                                     self._directional_difference,
                                     'zero',
                                     0,
                                     False
                                    )

    @cached_property
    def a1(self) -> numpy.array:
        return _directional_moment( self.variance_density.filled(0),
                                    self.radian_direction,
                                    self._directional_difference,
                                    'a',
                                    1,
                                    True
        )

    @cached_property
    def b1(self) -> numpy.array:
        return _directional_moment(self.variance_density.filled(0),
                                   self.radian_direction,
                                   self._directional_difference,
                                   'b',
                                   1,
                                   True
        )


    @cached_property
    def a2(self) -> numpy.array:
        return _directional_moment( self.variance_density.filled(0),
                                    self.radian_direction,
                                    self._directional_difference,
                                    'a',
                                    2,
                                    True
        )

    @cached_property
    def b2(self) -> numpy.array:
        return _directional_moment(self.variance_density.filled(0),
                                   self.radian_direction,
                                   self._directional_difference,
                                   'b',
                                   2,
                                   True
                                   )


    @WaveSpectrum.variance_density.setter
    def variance_density(self,val:numpy.ndarray):
        val = val.copy()
        mask = val < 0
        self._variance_density = MaskedArray(val,mask=mask,dtype='float64')
        self._update()

    def _update(self):
        super(WaveSpectrum2D, self)._update()

        if hasattr(self,'e'): delattr(self,'e')
        if hasattr(self,'a1'):    delattr(self,'a1')
        if hasattr(self,'b1'):    delattr(self,'b1')
        if hasattr(self,'a2'):    delattr(self,'a2')
        if hasattr(self,'b2'):    delattr(self,'b2')

        #self._e = None

    def _delta(self):
        angles = self.direction
        forward_diff = (numpy.diff(angles, append=angles[0]) + 180) % 360 - 180
        backward_diff = (numpy.diff(angles,
                                    prepend=angles[-1]) + 180) % 360 - 180
        return (forward_diff + backward_diff) / 2


    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> float:
        range = self._range(fmin, fmax)
        return numpy.trapz(self.e[range] * self.frequency[range] ** power,
                           self.frequency[range])

    def _create_wave_spectrum_input(self)-> dict:
        data = self.variance_density.filled(-1)
        return {
            "frequency":self.frequency,
            "varianceDensity":self.variance_density,
            "timestamp":self.timestamp,
            "latitude":self.latitude,
            "longitude":self.longitude
        }

    def copy(self)->"WaveSpectrum2D":
        return WaveSpectrum2D(
            frequency=self.frequency.copy(),
            directions=self.direction.copy(),
            varianceDensity=self.variance_density.copy(),
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude)

    def __add__(self, other:"WaveSpectrum2D")->"WaveSpectrum2D":
        spectrum = self.copy()

        self_dens = numpy.ma.MaskedArray(spectrum.variance_density)
        other_dens = numpy.ma.MaskedArray(other.variance_density)

        mask = self_dens.mask & other_dens.mask

        spectrum.variance_density = numpy.ma.MaskedArray(self_dens.filled(0) +
            other_dens.filled(0), mask)
        return spectrum

    def __sub__(self, other: "WaveSpectrum2D") -> "WaveSpectrum2D":
        spectrum = self.copy()
        self_dens = numpy.ma.MaskedArray(spectrum.variance_density)
        other_dens = numpy.ma.MaskedArray(other.variance_density)

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
    return WaveSpectrum2D(
        frequency=spectrum.frequency.copy(),
        varianceDensity=numpy.zeros_like(spectrum.variance_density),
        timestamp=spectrum.timestamp,
        latitude=spectrum.latitude,
        longitude=spectrum.longitude,
        directions=spectrum.direction.copy()
    )


@njit(cache=True)
def _directional_moment(
        density:numpy.ndarray,
        radian_direction:numpy.ndarray,
        directional_difference:numpy.ndarray,
        kind:str='zero',
        order:int=0,
        normalized:bool=True) -> numpy.array:

    """
    Calculate the directional moments of a directional Spectrum

    :param density: 2D variance density
    :param radian_direction: radian angles
    :param directional_difference: centered difference between directions
    :param kind: which Fourier moment to calculate: a=cosine, b=sine
    :param order: which order of Fourier moment to calculate
    :param normalized: whether to return normalized or non-normalized moments
    :return:
    """

    if kind == 'a':
        harmonic = numpy.cos(radian_direction * order) * directional_difference
    elif kind == 'b':
        harmonic = numpy.sin(radian_direction * order) * directional_difference
    elif kind == 'zero':
        harmonic = directional_difference
    else:
        raise Exception('Unknown moment')

    values = numpy.sum(density * harmonic, axis=-1)

    if normalized:
        scale = numpy.sum(density * directional_difference, axis=-1)
        scale[scale == 0] = numpy.float64(1.0)
    else:
        scale = numpy.ones( values.shape[0] )
    return values / scale
