import numpy
from scipy.optimize import minimize
from scipy.optimize import root
import typing
from numba import njit
from .utils import get_rhs,get_constraint_matrix,get_direction_increment

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

def mem2_optimize(directions_radians: numpy.ndarray,
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


    a1 = numpy.atleast_1d(a1)
    b1 = numpy.atleast_1d(b1)
    a2 = numpy.atleast_1d(a2)
    b2 = numpy.atleast_1d(b2)

    number_of_frequencies = len(a1)
    directional_distribution = numpy.zeros(
        (number_of_frequencies, len(directions_radians)))

    direction_increment = get_direction_increment(directions_radians)
    maximum_value = 1 / direction_increment.min()
    bounds = [(1e-10, maximum_value) for _ in
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
            entrophy_cost,
            guess,
            args=direction_increment,
            method='SLSQP',
            jac=entrophy_cost_jacobian,
            bounds=bounds,
            constraints=constraints
        )
        directional_distribution[ifreq, :] = res.x
    return numpy.squeeze(directional_distribution)


@njit()
def iter_func(lambdas, sine_and_cosine, a1, b1, a2, b2,
              direction_increment):
    vect = numpy.zeros(sine_and_cosine.shape[1])
    for jj in range(0, 4):
        vect = vect + lambdas[jj] * sine_and_cosine[jj, :]
    # dist = numpy.exp(- numpy.sum(  lambdas*sine_and_cosine,axis=0))
    dist = numpy.exp(- vect)
    out = numpy.zeros(4)
    out[0] = numpy.sum(
        (a1 - sine_and_cosine[0, :]) * dist * direction_increment
    )
    out[1] = numpy.sum(
        (b1 - sine_and_cosine[1, :]) * dist * direction_increment
    )
    out[2] = numpy.sum(
        (a2 - sine_and_cosine[2, :]) * dist * direction_increment
    )
    out[3] = numpy.sum(
        (b2 - sine_and_cosine[3, :]) * dist * direction_increment)
    return out


def reconstruction(
        lambdas,
        direction_increment,
        sine_and_cosine
):
    # sum_vector = numpy.sum(  lambdas[:,None]*sine_and_cosine,axis=0)
    sum_vector = numpy.zeros(sine_and_cosine.shape[1])
    for jj in range(0, 4):
        sum_vector = sum_vector + lambdas[jj] * sine_and_cosine[jj, :]

    lambda0 = numpy.log(
        numpy.sum(numpy.exp(-sum_vector) * direction_increment)
    )

    return numpy.exp(
        - (sum_vector + lambda0)
    )


def mem2(directions_radians: numpy.ndarray,
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

    def initial_value(a1, b1, a2, b2):
        guess = numpy.empty((len(a1), 4))
        fac = 1 + a1 ** 2 + b1 ** 2 + a2 ** 2 + b2 ** 2
        guess[:, 0] = (2 * a1 * a2 + 2 * b1 * b2 - 2 * a1 * fac)
        guess[:, 1] = (2 * a1 * b2 - 2 * b1 * a2 - 2 * b1 * fac)
        guess[:, 2] = (a1 ** 2 - b1 ** 2 - 2 * a2 * fac)
        guess[:, 3] = (2 * a1 * b1 - 2 * b2 * fac)
        return guess

    a1 = numpy.atleast_1d(a1)
    b1 = numpy.atleast_1d(b1)
    a2 = numpy.atleast_1d(a2)
    b2 = numpy.atleast_1d(b2)

    number_of_frequencies = len(a1)
    directional_distribution = numpy.zeros(
        (number_of_frequencies, len(directions_radians)))

    direction_increment = get_direction_increment(directions_radians)

    sine_and_cosine = numpy.empty((4, len(directions_radians)))
    sine_and_cosine[0, :] = numpy.cos(directions_radians)
    sine_and_cosine[1, :] = numpy.sin(directions_radians)
    sine_and_cosine[2, :] = numpy.cos(2 * directions_radians)
    sine_and_cosine[3, :] = numpy.sin(2 * directions_radians)

    guess = initial_value(a1, b1, a2, b2)
    for ifreq in range(0, number_of_frequencies):
        #
        res = root(
            iter_func,
            guess[ifreq, :],
            args=(
                sine_and_cosine, a1[ifreq], b1[ifreq], a2[ifreq], b2[ifreq],
                direction_increment)
        )
        lambas = res.x
        directional_distribution[ifreq, :] = reconstruction(
            lambas, direction_increment, sine_and_cosine)
    return numpy.squeeze(directional_distribution)