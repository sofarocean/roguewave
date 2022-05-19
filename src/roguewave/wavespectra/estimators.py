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
from scipy.ndimage import gaussian_filter

from scipy.optimize import minimize
from qpsolvers import solve_ls
import typing
from numba import njit

from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput


def mem(directions_radians: numpy.ndarray, a1: numpy.ndarray,
        b1: numpy.ndarray, a2: numpy.ndarray,
        b2: numpy.ndarray) -> numpy.ndarray:
    """
    This function uses the maximum entropy method by Lygre and Krogstadt (1986,JPO)
    to estimate the directional shape of the spectrum. Enthropy is defined in the
    Boltzmann sense (log D)

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the directional
    distribution in ocean wave spectra. Journal of Physical Oceanography, 16(12), 2052-2060.

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]. (going to, anti-clockswise from east)

    :param a1: 1d array of cosine directional moment as function of frequency,
    length [number_of_frequencies]

    :param b1: 1d array of sine directional moment as function of frequency,
    length [number_of_frequencies]

    :param a2: 1d array of double angle cosine directional moment as function
    of frequency, length [number_of_frequencies]

    :param b2: 1d array of double angle sine directional moment as function of
    frequency, length [number_of_frequencies]

    :return: array with shape [number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.

    Maximize the enthrophy of the solution with entrophy defined as:

           integrate log(D) over directions

    such that the resulting distribution D reproduces the observed moments.

    :return: Directional distribution as a numpy array

    Note that:
    d1 = a1; d2 =b1; d3 = a2 and d4=b2 in the defining equations 10.
    """

    # Ensure that these are numpy arrays
    a1 = numpy.atleast_1d(a1)
    b1 = numpy.atleast_1d(b1)
    a2 = numpy.atleast_1d(a2)
    b2 = numpy.atleast_1d(b2)

    number_of_directions = len(directions_radians)

    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2
    #
    # Eq. 13 L&K86
    #
    Phi1 = (c1 - c2 * numpy.conj(c1)) / (1 - c1 * numpy.conj(c1))
    Phi2 = c2 - Phi1 * c1
    #
    e1 = numpy.exp(-directions_radians * 1j)
    e2 = numpy.exp(-directions_radians * 2j)

    numerator = (1 - Phi1 * numpy.conj(c1) - Phi2 * numpy.conj(c2))
    denominator = numpy.abs(1 - Phi1[:, None] * e1[None, :]
                            - Phi2[:, None] * e2[None, :]) ** 2

    D = numpy.real(numerator[:, None] / denominator) / numpy.pi / 2

    # Normalize to 1. in discrete sense
    integralApprox = numpy.sum(D,
                               axis=-1) * numpy.pi * 2. / number_of_directions
    D = D / integralApprox[:, None]

    return numpy.squeeze(D)


def get_direction_increment(
        directions_radians: numpy.ndarray) -> numpy.ndarray:
    """
    calculate the stepsize used for midpoint integration. The directions
    represent the center of the interval - and we want to find the dimensions of
    the interval (difference between the preceeding and succsesive midpoint).

    :param directions_radians: array of radian directions
    :return: array of radian intervals
    """

    # Calculate the forward difference appending the first entry to the back
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    forward_diff = (numpy.diff(directions_radians,
                               append=directions_radians[0]) + numpy.pi) % (
                           2 * numpy.pi) - numpy.pi

    # Calculate the backward difference prepending the last entry to the front
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    backward_diff = (numpy.diff(directions_radians,
                                prepend=directions_radians[-1]) + numpy.pi) % (
                            2 * numpy.pi) - numpy.pi

    # The interval we are interested in is the average of the forward and backward
    # differences.
    return (forward_diff + backward_diff) / 2


def get_constraint_matrix(directions_radians: numpy.ndarray) -> numpy.ndarray:
    """
    Define the matrix M that can be used in the matrix product M@D (with D the
    directional distribution) such that:

            M@D = [1,a1,b1,a2,b2]^T

    with a1,b1 etc the directional moments at a given frequency.

    :param directions_radians: array of radian directions
    :return:
    """
    number_of_dir = len(directions_radians)
    constraints = numpy.zeros((5, number_of_dir))
    direction_increment = get_direction_increment(directions_radians)
    constraints[0, :] = direction_increment
    constraints[1, :] = direction_increment * numpy.cos(directions_radians)
    constraints[2, :] = direction_increment * numpy.sin(directions_radians)
    constraints[3, :] = direction_increment * numpy.cos(2 * directions_radians)
    constraints[4, :] = direction_increment * numpy.sin(2 * directions_radians)
    return constraints


def get_rhs(a1: numpy.ndarray, b1: numpy.ndarray, a2: numpy.ndarray,
            b2: numpy.ndarray) -> numpy.ndarray:
    """
    Define the matrix rhs that for each row contains the directional moments
    at a given frequency:

    rhs = [ 1, a1[0],b1[0],a2[0],b2[0],
            |    |    |      |    |
            N, a1[0],b1[0],a2[0],b2[0] ]

    These rows are use as the "right hand side" in the linear constraints
    (see get_constraint_matrix)

    :param a1: 1d array of cosine directional moment as function of frequency,
    length [number_of_frequencies]

    :param b1: 1d array of sine directional moment as function of frequency,
    length [number_of_frequencies]

    :param a2: 1d array of double angle cosine directional moment as function
    of frequency, length [number_of_frequencies]

    :param b2: 1d array of double angle sine directional moment as function of
    frequency, length [number_of_frequencies]

    :return: array ( number of frequencies by 5) that for each row contains
    the directional moments at a given frequency
    """
    rhs = numpy.array([numpy.ones_like(a1), a1, b1, a2, b2]).transpose()
    return rhs


def maximize_shannon_entrophy(directions_radians: numpy.ndarray,
                              a1: typing.Union[numpy.ndarray, float],
                              b1: typing.Union[numpy.ndarray, float],
                              a2: typing.Union[numpy.ndarray, float],
                              b2: typing.Union[
                                  numpy.ndarray, float]) -> numpy.ndarray:
    """
    Return the directional distribution that maximizes Shannon [ - D log(D) ]
    enthrophy constrained by given observed directional moments,

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

    :param a1: 1d array of cosine directional moment as function of frequency,
    length [number_of_frequencies]

    :param b1: 1d array of sine directional moment as function of frequency,
    length [number_of_frequencies]

    :param a2: 1d array of double angle cosine directional moment as function
    of frequency, length [number_of_frequencies]

    :param b2: 1d array of double angle sine directional moment as function of
    frequency, length [number_of_frequencies]

    :return: array with shape [number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.

    Maximize the enthrophy of the solution with entrophy defined as:

           integrate - D * log(D) over directions

    such that the resulting distribution D reproduces the observed moments.

    Implementation notes:
    - we use the scipy opmtimize minimize method with SLSQP to solve the
      constrained optimization problem.
    - we use numba for the cost function, constraints and their Jacobians to
      speed the operations of the minimizer.
    """

    @njit()
    def entrophy_cost(D, delta):
        return numpy.sum(delta * D * numpy.log(D))

    @njit()
    def entrophy_cost_jacobian(D, delta):
        return delta * numpy.log(D) + delta

    @njit()
    def constraints_function(D, constraint_matrix, rhs):
        return constraint_matrix @ D - rhs

    @njit()
    def constraints_jacobian_function(D, constraint_matrix, rhs):
        return constraint_matrix

    a1 = numpy.atleast_1d(a1)
    b1 = numpy.atleast_1d(b1)
    a2 = numpy.atleast_1d(a2)
    b2 = numpy.atleast_1d(b2)

    number_of_frequencies = len(a1)
    directional_distribution = numpy.zeros(
        (number_of_frequencies, len(directions_radians)))

    direction_increment = get_direction_increment(directions_radians)
    maximum_value = 1 / direction_increment.min()
    bounds = [(1e-10, maximum_value) for ii in
              range(0, len(directions_radians))]
    guess = numpy.ones_like(directions_radians) / 360

    constraint_matrix = get_constraint_matrix(directions_radians)
    rhs = get_rhs(a1, b1, a2, b2)

    for ifreq in range(0, number_of_frequencies):
        constraints = [
            {
                'type': 'eq',
                'fun': constraints_function,
                'jac': constraints_jacobian_function,
                'args': (constraint_matrix, rhs[ifreq, :])
            }
        ]

        res = minimize(
            entrophy_cost, guess, args=direction_increment, method='SLSQP',
            jac=entrophy_cost_jacobian, bounds=bounds,
            constraints=constraints
        )

        directional_distribution[ifreq, :] = res.x
    return numpy.squeeze(directional_distribution)


def ridge_regression(directions_radians: numpy.ndarray, a1, b1, a2,
                     b2) -> numpy.ndarray:
    """
    Return the directional distribution that minimizes the variance (D**2)
    constrained by given observed directional moments,

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

    :param a1: 1d array of cosine directional moment as function of frequency,
    length [number_of_frequencies]

    :param b1: 1d array of sine directional moment as function of frequency,
    length [number_of_frequencies]

    :param a2: 1d array of double angle cosine directional moment as function
    of frequency, length [number_of_frequencies]

    :param b2: 1d array of double angle sine directional moment as function of
    frequency, length [number_of_frequencies]

    :return: array with shape [number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.

    Minimize the variance of the solution:

           integrate D**2 over directions

    such that the resulting distribution D reproduces the observed moments.

    Implementation notes:
    - we formulate the problem as a standard Quadratic Programming problem which
      can them be solved efficiently with the qpsolvers package.
    """

    a1 = numpy.atleast_1d(a1)
    b1 = numpy.atleast_1d(b1)
    a2 = numpy.atleast_1d(a2)
    b2 = numpy.atleast_1d(b2)

    number_of_frequencies = len(a1)
    directional_distribution = numpy.zeros(
        (number_of_frequencies, len(directions_radians)))

    constraint_matrix = get_constraint_matrix(directions_radians)
    rhs = get_rhs(a1, b1, a2, b2)
    identity_matrix = numpy.diag(numpy.ones_like(directions_radians), 0)

    zeros = numpy.zeros_like(directions_radians)
    direction_increment = get_direction_increment(directions_radians)
    upperbound = numpy.ones_like(
        directions_radians) / direction_increment.min()

    for ifreq in range(0, number_of_frequencies):
        res = solve_ls(R=identity_matrix, s=zeros,
                       # minimizing |Rx-b|**2
                       lb=zeros, ub=upperbound,
                       # lb: non-negative; ub: binwidth * ub = 1
                       A=constraint_matrix, b=rhs[ifreq, :], verbose=False
                       # with hard constraint that Ax=b
                       )

        if res is None:
            raise Exception('No solution')

        directional_distribution[ifreq, :] = res

    return numpy.squeeze(directional_distribution)


def spect2d_from_spec1d(spectrum1D: WaveSpectrum1D,
                        number_of_directions: int = 36,
                        method: str = 'ridge', frequency_smoothing=True,
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
    elif method.lower() in ['ridge_regression', 'ridge']:
        directional_distribution = ridge_regression(direction * numpy.pi / 180,
                                                    a1, b1, a2,
                                                    b2) * Jacobian
    elif method.lower() in ['maximum_shannonw_entrophy_method', 'msem']:
        directional_distribution = maximize_shannon_entrophy(
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

def spec1d_from_spec2d(spectrum:WaveSpectrum2D)->WaveSpectrum1D:
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