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
        self.direction = numpy.array(wave_spectrum2D_input['directions'],dtype='float64')
        self._frequency_peak_indices = None
        self._direction_peak_indices = None
        self._update()

    @WaveSpectrum.variance_density.setter
    def variance_density(self,val:numpy.ndarray):
        mask = val < 0
        self._variance_density = MaskedArray(val,mask=mask,dtype='float64')
        self._update()

    def _update(self):
        if self.direction is not None:
            self._a1 = self._directional_moment('a', 1, normalized=True)
            self._b1 = self._directional_moment('b', 1, normalized=True)
            self._a2 = self._directional_moment('a', 2, normalized=True)
            self._b2 = self._directional_moment('b', 2, normalized=True)
            self._e = self._directional_moment('zero', 0, normalized=False)
            self._frequency_peak_indices, self._direction_peak_indices = self.find_peak_indices(update=True)

    def _delta(self):
        angles = self.direction
        forward_diff = (numpy.diff(angles, append=angles[0]) + 180) % 360 - 180
        backward_diff = (numpy.diff(angles,
                                    prepend=angles[-1]) + 180) % 360 - 180
        return (forward_diff + backward_diff) / 2

    def _directional_moment(self, kind='zero', order=0,
                            normalized=True) -> numpy.array:
        delta = self._delta()
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

    def extract(self,mask:numpy.ndarray, minimum_ratio=None)->"WaveSpectrum2D":
        spectrum = self.copy()

        if minimum_ratio:
            ii,jj = self.find_peak_indices()
            peak_density = self._variance_density[ ii[0], jj[0] ]
            if peak_density > 0.0:
                mask = mask & (spectrum.variance_density/peak_density > minimum_ratio)
            else:
                print('eh')
        density = numpy.ma.masked_array( spectrum.variance_density, mask=~mask)
        spectrum.variance_density = density
        return spectrum

    def peak_direction(self):
        _, index = self.find_peak_indices()
        return self.direction[index[0]]

    def peak_wavenumber(self, depth=numpy.inf):
        index, _ = self.find_peak_indices()

        return inverse_intrinsic_dispersion_relation(
            self.radian_frequency[index[0]], depth)

    def peak_wavenumber_east(self, depth=numpy.inf):
        wave_number    = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi/180
        return wave_number * numpy.cos(wave_direction)

    def peak_wavenumber_north(self, depth=numpy.inf):
        wave_number    = self.peak_wavenumber()
        wave_direction = self.peak_direction() * numpy.pi/180
        return wave_number * numpy.cos(wave_direction)

    def find_peak_indices(self,update=False):
        """
        Find the peaks of the frequency-direction spectrum.
        :param density: 2D variance density
        :return: List of indices indicating the maximum
        """
        if not update and self._frequency_peak_indices is not None:
            return self._frequency_peak_indices, self._direction_peak_indices

        density = self.variance_density

        # create a search region
        neighborhood = scipy.ndimage.generate_binary_structure(density.ndim, 2)

        # Set the local neighbourhood to the the maximum value, use wrap.
        # Note that technically frequency wrapping is incorrect- but unlikely to
        # cause issues. First we apply the maximum filter .with 9 point footprint
        # to set each pixel of the output to the local maximum
        filtered = scipy.ndimage.maximum_filter(
            density, footprint=neighborhood,mode='wrap'
        )

        # Then we find maxima based on equality with input array
        maximum_mask = filtered == density

        # Remove possible background (0's)
        background_mask = density == 0

        # Remove peaks in background
        maximum_mask[background_mask] = False

        # return indices of maxima
        ii, jj = numpy.where(maximum_mask)

        # sort from lowest maximum value to largest maximum value.
        sorted = numpy.flip(numpy.argsort(density[ii, jj]))
        ii = ii[sorted]
        jj = jj[sorted]
        if len(ii) == 0:
            ii = numpy.array([0])
            jj = numpy.array([0])
        self._frequency_peak_indices = ii
        self._direction_peak_indices = jj
        return ii,jj


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




