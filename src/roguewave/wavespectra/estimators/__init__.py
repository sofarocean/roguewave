"""
Contents: Spectral estimators

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Spectral estimators that can be used to create a 2D spectrum from buoy
observations

Classes:

- `None

Functions:

- `mem`, maximum entrophy method
- `mem2`, ...

How To Use This Module
======================
(See the individual functions for details.)

1. Import it: ``import estimators`` or ``from estimators import ...``.
2. call spec2d_from_spec1d to transform a 1D spectrum into a 2D spectrum.
"""

import numpy
from .mem2 import mem2
from .mem import mem
from .loglikelyhood import log_likelyhood
from scipy.ndimage import gaussian_filter
from typing import overload, Dict, List
from roguewave.wavespectra.spectrum2D import WaveSpectrum2D
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D


# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
@overload
def convert_to_2d_spectrum(data: Dict[str, list[WaveSpectrum1D]],
                           number_of_directions: int = 36,
                           method: str = 'mem2', frequency_smoothing=False,
                           smoothing_lengthscale=1) -> Dict[
    str, list[WaveSpectrum2D]]: ...


@overload
def convert_to_2d_spectrum(data: list[WaveSpectrum1D],
                           number_of_directions: int = 36,
                           method: str = 'mem2', frequency_smoothing=False,
                           smoothing_lengthscale=1) -> list[
    WaveSpectrum2D]: ...


@overload
def convert_to_2d_spectrum(data: WaveSpectrum1D,
                           number_of_directions: int = 36,
                           method: str = 'mem2', frequency_smoothing=False,
                           smoothing_lengthscale=1) -> WaveSpectrum2D: ...


@overload
def convert_to_1d_spectrum(data: Dict[str, list[WaveSpectrum2D]]) -> Dict[
    str, list[WaveSpectrum1D]]: ...


@overload
def convert_to_1d_spectrum(data: list[WaveSpectrum2D]) -> list[
    WaveSpectrum1D]: ...


@overload
def convert_to_1d_spectrum(data: WaveSpectrum2D) -> WaveSpectrum1D: ...


# -----------------------------------------------------------------------------
#                              Implementation
# -----------------------------------------------------------------------------
def convert_to_2d_spectrum(data, number_of_directions: int = 36,
                           method: str = 'mem2', frequency_smoothing=False,
                           smoothing_lengthscale=1):
    if isinstance(data, dict):
        out = {}
        for key in data:
            out[key] = convert_to_2d_spectrum(data[key], number_of_directions,
                                              method, frequency_smoothing,
                                              smoothing_lengthscale)
        return out
    elif isinstance(data, list):
        out = []
        for item in data:
            out.append(convert_to_2d_spectrum(item, number_of_directions,
                                              method, frequency_smoothing,
                                              smoothing_lengthscale))
        return out
    elif isinstance(data, WaveSpectrum1D):
        return spec2d_from_spec1d(data,number_of_directions,
                                              method, frequency_smoothing,
                                              smoothing_lengthscale)
    elif isinstance(data, WaveSpectrum2D):
        return data
    else:
        raise Exception('Cannot convert to 2D spectrum')


def convert_to_1d_spectrum(data):
    if isinstance(data, dict):
        out = {}
        for key in data:
            out[key] = convert_to_1d_spectrum(data[key])
        return out
    elif isinstance(data, list):
        out = []
        for item in data:
            out.append(convert_to_1d_spectrum(item))
        return out
    elif isinstance(data, WaveSpectrum1D):
        return data
    elif isinstance(data, WaveSpectrum2D):
        return spec1d_from_spec2d(data)
    else:
        raise Exception('Cannot convert to 2D spectrum')


def spec2d_from_spec1d(spectrum1D: WaveSpectrum1D,
                       number_of_directions: int = 36,
                       method: str = 'mem2', frequency_smoothing=False,
                       smoothing_lengthscale=1) -> WaveSpectrum2D:
    """
    Construct a 2D spectrum based on the 1.5D spectrum and a spectral
    reconstruction method.

    :param number_of_directions: length of the directional vector for the
    2D spectrum. Directions returned are in degrees

    :param method: Choose a method in ['mem','ridge','msem']
        mem: maximum entrophy (in the Boltzmann sense) method
        Lygre, A., & Krogstad, H. E. (1986). Explicit expression and
        fast but tends to create narrow spectra anderroneous secondary peaks.

        mem2: use entrophy (in the Shannon sense) to maximize. Likely
        best method see- Benoit, M. (1993).

        log: solve underdetermined constrained problem using ridge regression
        as regulizer. Performant because efficient solutions to constrained
        quadratic programming problems exist. The spectra compare well to
        nlmen- though are somewhat broader in general. Does not suffer from
        the spurious peak issue of mem. Default method for there reasons.

    REFERENCES:
    Benoit, M. (1993). Practical comparative performance survey of methods
        used for estimating directional wave spectra from heave-pitch-roll data.
        In Coastal Engineering 1992 (pp. 62-75).

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
        directional distribution in ocean wave spectra.
        Journal of Physical Oceanography, 16(12), 2052-2060.

    """
    direction = numpy.linspace(0, 360, number_of_directions,
                               endpoint=False)

    # Jacobian to transform distribution as function of radian angles into
    # degrees.
    Jacobian = numpy.pi / 180

    a1 = spectrum1D.a1
    b1 = spectrum1D.b1
    a2 = spectrum1D.a2
    b2 = spectrum1D.b2
    e = spectrum1D.e
    if frequency_smoothing:
        e = gaussian_filter(spectrum1D.e, smoothing_lengthscale)
        a1 = gaussian_filter(a1 * spectrum1D.e, smoothing_lengthscale) / e
        a2 = gaussian_filter(a2 * spectrum1D.e, smoothing_lengthscale) / e
        b1 = gaussian_filter(b1 * spectrum1D.e, smoothing_lengthscale) / e
        b2 = gaussian_filter(b2 * spectrum1D.e, smoothing_lengthscale) / e

        scale = spectrum1D.m0() / numpy.trapz(e, spectrum1D.frequency)
        e = e * scale

    if method.lower() in ['maximum_entropy_method', 'mem']:
        # reconstruct the directional distribution using the maximum entropy
        # method.
        directional_distribution = mem(direction * numpy.pi / 180, a1, b1, a2,
                                       b2) * Jacobian
    elif method.lower() in ['log_likelyhood', 'log']:
        directional_distribution = log_likelyhood(direction * numpy.pi / 180,
                                                  a1, b1, a2,
                                                  b2) * Jacobian
    elif method.lower() in ['maximum_entrophy_method2', 'mem2']:
        directional_distribution = mem2(
            direction * numpy.pi / 180, a1, b1, a2,
            b2) * Jacobian
    else:
        raise Exception(f'unsupported spectral estimator method: {method}')

    # We return a 2D wave spectrum object.
    return WaveSpectrum2D(
        frequency=spectrum1D.frequency,
        directions=direction,
        varianceDensity=e[:, None] * directional_distribution,
        timestamp=spectrum1D.timestamp,
        longitude=spectrum1D.longitude,
        latitude=spectrum1D.latitude
    )


def spec1d_from_spec2d(spectrum: WaveSpectrum2D) -> WaveSpectrum1D:
    return WaveSpectrum1D(
        frequency=spectrum.frequency,
        varianceDensity=spectrum.e,
        timestamp=spectrum.timestamp,
        longitude=spectrum.longitude,
        latitude=spectrum.latitude,
        a1=spectrum.a1,
        b1=spectrum.b1,
        a2=spectrum.a2,
        b2=spectrum.b2,
    )
