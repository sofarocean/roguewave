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
"""
from roguewave.wavespectra.estimators.mem2 import mem2
from roguewave.wavespectra.estimators.mem import mem
from roguewave.wavespectra.estimators.loglikelyhood import log_likelyhood
import numpy

# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
def spec2d_from_spec1d(
        e:numpy.ndarray,
        a1:numpy.ndarray,
        b1:numpy.ndarray,
        a2:numpy.ndarray,
        b2:numpy.ndarray,
        direction:numpy.ndarray,
        method: str = 'mem2') -> numpy.ndarray:
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


    # Jacobian to transform distribution as function of radian angles into
    # degrees.
    Jacobian = numpy.pi / 180

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
    return e[:, None] * directional_distribution
