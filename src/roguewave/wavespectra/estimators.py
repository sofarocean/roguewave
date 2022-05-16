"""
Contents: Spectral estimators

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""

import numpy

from scipy.optimize import minimize
from qpsolvers import solve_ls
import typing
from numba import njit


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


def get_direction_increment(directions_radians):
    forward_diff = (numpy.diff(directions_radians,
                               append=directions_radians[0]) + numpy.pi) % (
                           2 * numpy.pi) - numpy.pi

    backward_diff = (numpy.diff(directions_radians,
                                prepend=directions_radians[-1]) + numpy.pi) % (
                            2 * numpy.pi) - numpy.pi

    return (forward_diff + backward_diff) / 2


def get_constraint_matrix(directions_radians):
    number_of_dir = len(directions_radians)
    constraints = numpy.zeros((5, number_of_dir))
    direction_increment = get_direction_increment(directions_radians)
    constraints[0, :] = direction_increment
    constraints[1, :] = direction_increment * numpy.cos(directions_radians)
    constraints[2, :] = direction_increment * numpy.sin(directions_radians)
    constraints[3, :] = direction_increment * numpy.cos(2 * directions_radians)
    constraints[4, :] = direction_increment * numpy.sin(2 * directions_radians)
    return constraints


def get_rhs(a1, b1, a2, b2):
    rhs = numpy.array([numpy.ones_like(a1), a1, b1, a2, b2]).transpose()
    return rhs


def maximize_entrophy(directions_radians: numpy.ndarray,
                      a1: typing.Union[numpy.ndarray, float],
                      b1: typing.Union[numpy.ndarray, float],
                      a2: typing.Union[numpy.ndarray, float],
                      b2: typing.Union[numpy.ndarray, float]) -> numpy.ndarray:
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
