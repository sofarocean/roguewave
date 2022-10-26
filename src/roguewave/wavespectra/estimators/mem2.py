"""
Implementation of the "MEM2" method:

see Kim1995:

    Kim, T., Lin, L. H., & Wang, H. (1995). Application of maximum entropy method
    to the real sea data. In Coastal Engineering 1994 (pp. 340-355).

    link: https://icce-ojs-tamu.tdl.org/icce/index.php/icce/article/download/4967/4647
    (working as of May 29, 2022)

and references therein.

"""
import numpy
from scipy.optimize import root
import typing
from roguewave.wavespectra.estimators.utils import get_direction_increment
from numba import njit, generated_jit, types, prange


def mem2(
    directions_radians: numpy.ndarray,
    a1: typing.Union[numpy.ndarray, float],
    b1: typing.Union[numpy.ndarray, float],
    a2: typing.Union[numpy.ndarray, float],
    b2: typing.Union[numpy.ndarray, float],
    progress,
    solution_method="newton",
) -> numpy.ndarray:
    """

    :param directions_radians:
    :param a1:
    :param b1:
    :param a2:
    :param b2:
    :param solution_method:
    :return:
    """

    if solution_method == "scipy":
        func = mem2_scipy_root_finder
        kwargs = {}

    elif solution_method == "newton":
        func = mem2_newton
        kwargs = {}

    elif solution_method == "approximate":
        func = mem2_newton
        kwargs = {"approximate": True}

    else:
        raise ValueError("Unknown method")

    return func(directions_radians, a1, b1, a2, b2, progress, **kwargs)


@njit(cache=True)
def moment_constraints(lambdas, twiddle_factors, moments, direction_increment):
    """
    Construct the nonlinear equations we need to solve for lambda.

    :param lambdas:
    :param twiddle_factors:
    :param moments
    :param direction_increment:
    :return:
    """

    dist = mem2_directional_distribution(lambdas, direction_increment, twiddle_factors)
    out = numpy.zeros(4)
    for mm in range(0, 4):
        out[mm] = moments[mm] - numpy.sum(
            (twiddle_factors[mm, :]) * dist * direction_increment
        )
    return out


@njit(cache=True)
def _jacobian(lagrange_multiplier, twiddle_factors, direction_increment):
    """
    Calculate the jacobian of the constraint equations.

    :param lagrange_multiplier:
    :param twiddle_factors:
    :param direction_increment:
    :return:
    """
    inner_product = numpy.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    # We subtract the minimum to ensure that the values in the exponent do not become too large. This amounts to
    # multiplyig with a constant - which is fine since we normalize anyway. Effectively- this avoids overflow errors
    # (or infinities) - at the expense of underflowing (which is less of an issue).
    #
    inner_product = inner_product - numpy.min(inner_product)

    normalization = 1 / numpy.sum(numpy.exp(-inner_product) * direction_increment)
    shape = numpy.exp(-inner_product)

    normalization_derivative = numpy.zeros(4)
    for mm in range(0, 4):
        normalization_derivative[mm] = normalization * numpy.sum(
            twiddle_factors[mm, :] * numpy.exp(-inner_product) * direction_increment
        )

    # To note- we have to multiply seperately to avoid potential underflow/overflow errors.
    normalization_derivative = normalization_derivative * normalization

    shape_derivative = numpy.zeros((4, twiddle_factors.shape[1]))
    for mm in range(0, 4):
        shape_derivative[mm, :] = -twiddle_factors[mm, :] * shape

    jacobian = numpy.zeros((4, 4))
    for mm in range(0, 4):
        for nn in range(0, 4):
            jacobian[mm, nn] = -numpy.sum(
                twiddle_factors[mm, :]
                * direction_increment
                * (
                    normalization * shape_derivative[nn, :]
                    + shape * normalization_derivative[nn]
                ),
                -1,
            )
    return jacobian


@njit(cache=True)
def mem2_directional_distribution(
    lagrange_multiplier, direction_increment, twiddle_factors
):
    """
    Given the solution for the Lagrange multipliers- reconstruct the directional
    distribution.
    :param lagrange_multiplier:
    :param direction_increment:
    :param twiddle_factors:
    :return:
    """
    inner_product = numpy.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    inner_product = inner_product - numpy.min(inner_product)

    normalization = 1 / numpy.sum(numpy.exp(-inner_product) * direction_increment)
    return numpy.exp(-inner_product) * normalization


@njit(cache=True)
def initial_value(a1, b1, a2, b2):
    """
    Initial guess of the Lagrange Multipliers according to the "MEM AP2" approximation
    found im Kim1995

    :param a1:
    :param b1:
    :param a2:
    :param b2:
    :return:
    """
    guess = numpy.empty((*a1.shape, 4))
    fac = 1 + a1**2 + b1**2 + a2**2 + b2**2
    guess[..., 0] = 2 * a1 * a2 + 2 * b1 * b2 - 2 * a1 * fac
    guess[..., 1] = 2 * a1 * b2 - 2 * b1 * a2 - 2 * b1 * fac
    guess[..., 2] = a1**2 - b1**2 - 2 * a2 * fac
    guess[..., 3] = 2 * a1 * b1 - 2 * b2 * fac
    return guess


def mem2_scipy_root_finder(
    directions_radians: numpy.ndarray,
    a1: typing.Union[numpy.ndarray, float],
    b1: typing.Union[numpy.ndarray, float],
    a2: typing.Union[numpy.ndarray, float],
    b2: typing.Union[numpy.ndarray, float],
    progress,
    **kwargs
) -> numpy.ndarray:
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

    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = numpy.zeros(
        (number_of_points, number_of_frequencies, len(directions_radians))
    )

    direction_increment = get_direction_increment(directions_radians)

    twiddle_factors = numpy.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = numpy.cos(directions_radians)
    twiddle_factors[1, :] = numpy.sin(directions_radians)
    twiddle_factors[2, :] = numpy.cos(2 * directions_radians)
    twiddle_factors[3, :] = numpy.sin(2 * directions_radians)

    guess = initial_value(a1, b1, a2, b2)
    for ipoint in range(0, number_of_points):
        progress.update(1)
        for ifreq in range(0, number_of_frequencies):
            #
            moments = numpy.array(
                [
                    a1[ipoint, ifreq],
                    b1[ipoint, ifreq],
                    a2[ipoint, ifreq],
                    b2[ipoint, ifreq],
                ]
            )
            res = root(
                moment_constraints,
                guess[ipoint, ifreq, :],
                args=(twiddle_factors, moments, direction_increment),
                method="lm",
            )
            lambas = res.x

            directional_distribution[ipoint, ifreq, :] = mem2_directional_distribution(
                lambas, direction_increment, twiddle_factors
            )

    return directional_distribution


@njit(cache=True, nogil=True, parallel=False)
def mem2_newton(
    directions_radians: numpy.ndarray,
    a1: typing.Union[numpy.ndarray, float],
    b1: typing.Union[numpy.ndarray, float],
    a2: typing.Union[numpy.ndarray, float],
    b2: typing.Union[numpy.ndarray, float],
    progress,
    max_iter=100,
    atol=1e-2,
    rcond=1e-6,
    approximate=False,
) -> numpy.ndarray:
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

    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = numpy.zeros(
        (number_of_points, number_of_frequencies, len(directions_radians))
    )

    direction_increment_downward_difference = (
        directions_radians - numpy.roll(directions_radians, 1) + numpy.pi
    ) % (2 * numpy.pi) - numpy.pi
    direction_increment_upward_difference = (
        -(directions_radians - numpy.roll(directions_radians, -1) + numpy.pi)
        % (2 * numpy.pi)
        - numpy.pi
    )

    direction_increment = (
        direction_increment_downward_difference + direction_increment_upward_difference
    ) / 2

    twiddle_factors = numpy.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = numpy.cos(directions_radians)
    twiddle_factors[1, :] = numpy.sin(directions_radians)
    twiddle_factors[2, :] = numpy.cos(2 * directions_radians)
    twiddle_factors[3, :] = numpy.sin(2 * directions_radians)

    guess = initial_value(a1, b1, a2, b2)
    for ipoint in range(0, number_of_points):
        progress.update(1)
        _mem2_newton_point(
            directional_distribution[ipoint, :, :],
            number_of_frequencies,
            a1[ipoint, :],
            b1[ipoint, :],
            a2[ipoint, :],
            b2[ipoint, :],
            guess[ipoint, :, :],
            direction_increment,
            twiddle_factors,
            max_iter,
            rcond,
            atol,
            approximate,
        )

    return directional_distribution


@njit(cache=True, nogil=True, parallel=False)
def _mem2_newton_point(
    out,
    number_of_frequencies,
    a1,
    b1,
    a2,
    b2,
    guess,
    direction_increment,
    twiddle_factors,
    max_iter,
    rcond,
    atol,
    approximate,
):
    for ifreq in prange(0, number_of_frequencies):
        #
        moments = numpy.array([a1[ifreq], b1[ifreq], a2[ifreq], b2[ifreq]])
        out[ifreq, :] = estimate_distribution_newton(
            moments,
            guess[ifreq, :],
            direction_increment,
            twiddle_factors,
            max_iter,
            rcond,
            atol,
            approximate=approximate,
        )


@njit(cache=True)
def estimate_distribution_newton(
    moments: numpy.ndarray,
    guess: numpy.ndarray,
    direction_increment: numpy.ndarray,
    twiddle_factors: numpy.ndarray,
    max_iter: int = 100,
    rcond: float = 1e-3,
    atol: float = 1e-3,
    max_line_search_depth=8,
    approximate=False,
) -> numpy.ndarray:
    """
    Newton iteration to find the solution to the non-linear system of constraint equations defining the lagrange
    multipliers in the MEM2 method. Because the Lagrange multipliers enter the equations as exponents the system can
    be unstable to solve numerically.

    :param moments: the normalized directional moments [a1,b1,a2,b2]
    :param guess: first guess for the lagrange multipliers (ndarray, length 4)
    :param direction_increment: directional stepsize used in the integration, nd-array
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param max_iter: maximum number of iterations
    :param rcond: cut of value for singular values in the Jacobian matrix.
    :param atol: absolute accuracy of the moments.
    :return:
    """
    directional_distribution = numpy.empty(len(direction_increment))
    if numpy.any(numpy.isnan(guess)):
        directional_distribution[:] = 0
        return directional_distribution

    if approximate:
        directional_distribution[:] = mem2_directional_distribution(
            guess, direction_increment, twiddle_factors
        )
        return directional_distribution

    current_lagrange_multiplier_iterate = guess
    current_iterate_func_eval = moment_constraints(
        current_lagrange_multiplier_iterate,
        twiddle_factors,
        moments,
        direction_increment,
    )
    guess_func = current_iterate_func_eval

    magnitude_current_iterate = numpy.linalg.norm(current_lagrange_multiplier_iterate)
    magnitude_cur_func_eval = numpy.linalg.norm(current_iterate_func_eval)
    for iter in range(0, max_iter):
        if numpy.linalg.norm(current_iterate_func_eval) < atol:
            break
        #
        # Compute jacobian, and find newton iterate innovation as we solve for:
        #
        #       jacobian @ delta = - current_iterate_func_eval
        #
        # with:
        #
        #       delta = next_lagrange_multiplier_iterate-cur_lagrange_multiplier_iterate

        jacobian = _jacobian(
            current_lagrange_multiplier_iterate, twiddle_factors, direction_increment
        )
        lagrange_multiplier_delta = numpy.linalg.lstsq(
            jacobian, -current_iterate_func_eval, rcond=rcond
        )[0]
        # lagrange_multiplier_delta = numpy.linalg.solve(
        #     jacobian, -current_iterate_func_eval
        # )

        line_search_factor = 1

        magnitude_current_iterate = numpy.linalg.norm(
            current_lagrange_multiplier_iterate
        )
        magnitude_cur_func_eval = numpy.linalg.norm(current_iterate_func_eval)
        magnitude_update = numpy.linalg.norm(lagrange_multiplier_delta)
        inverse_relative_update = magnitude_current_iterate / magnitude_update

        # Do a line search for the optimum decrease. This is intended to stabilize the algorithm
        # as the equations are ill-posed.
        for ii in range(max_line_search_depth):
            update = line_search_factor * lagrange_multiplier_delta
            next_lagrange_multiplier_iterate = (
                current_lagrange_multiplier_iterate + update
            )

            next_iterate_func_eval = moment_constraints(
                next_lagrange_multiplier_iterate,
                twiddle_factors,
                moments,
                direction_increment,
            )
            magnitude_next_func_eval = numpy.linalg.norm(next_iterate_func_eval)

            if magnitude_next_func_eval <= magnitude_cur_func_eval:
                # If we are decreasing- continue
                current_iterate_func_eval = next_iterate_func_eval
                current_lagrange_multiplier_iterate = next_lagrange_multiplier_iterate
                break
            else:
                line_search_factor = min(
                    inverse_relative_update, line_search_factor / 2
                )

        else:
            # we are stuck - if lambdas get large we run into numerical issues
            if (
                magnitude_current_iterate > 100
                and magnitude_cur_func_eval < numpy.linalg.norm(guess_func)
            ):
                break
            else:
                raise ValueError("we did not converge")

    else:
        if (
            magnitude_current_iterate > 100
            and magnitude_cur_func_eval < numpy.linalg.norm(guess_func)
        ):
            pass
        else:
            raise ValueError("we did not converge")

    directional_distribution[:] = mem2_directional_distribution(
        current_lagrange_multiplier_iterate, direction_increment, twiddle_factors
    )

    return directional_distribution


@generated_jit(nopython=True)
def atleast_1d(x) -> numpy.ndarray:
    if x in types.number_domain:
        return lambda x: numpy.array([x])
    return lambda x: numpy.atleast_1d(x)
