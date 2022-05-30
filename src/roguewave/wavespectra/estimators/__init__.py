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
- `get_direction_increment`, calculate wrapped direction intervals for arbitrary increasing vector
- `get_constraint_matrix`, build the matrix that calculates moments from directions
- `maximize_shannon_entrophy`, estimate spectrum by maximizing Shannon entrophy
- `ridge_regression`, estimate spectrum by minimizing variance
- `spec2d_from_spec1d`, estimate the 2D spectrum from a given 1.5D spectrum using the desired method (default maximize_shannon_entrophy)

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

from roguewave.wavespectra.spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput

def spect2d_from_spec1d(spectrum1D: WaveSpectrum1D,
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

        msem: use entrophy (in the Shannon sense) to maximize. Likely
        best method see- Benoit, M. (1993). Is rather slow because we solve
        a general nonlinear constrained optimization problem.

        ridge: solve underdetermined constrained problem using ridge regression
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

    wave_spectrum2D_input = WaveSpectrum2DInput(
        frequency=spectrum1D.frequency,
        directions=direction,
        varianceDensity=e[:, None] * directional_distribution,
        timestamp=spectrum1D.timestamp,
        longitude=spectrum1D.longitude,
        latitude=spectrum1D.latitude
    )

    # We return a 2D wave spectrum object.
    return WaveSpectrum2D(wave_spectrum2D_input)


def spec1d_from_spec2d(spectrum: WaveSpectrum2D) -> WaveSpectrum1D:
    wave_spectrum1D_input = WaveSpectrum1DInput(
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
    return WaveSpectrum1D(wave_spectrum1D_input)





