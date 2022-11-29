from numba import njit
from numpy import (
    empty_like,
    arctan,
    tan,
    sqrt,
    cos,
    sin,
    pi,
    exp,
    zeros_like,
    abs,
    array,
    zeros,
)
from numpy.typing import NDArray

DEFAULT_ORDER = 4
DEFAULT_N = 1


@njit(cache=True)
def integrate(
    time: NDArray, signal: NDArray, order=DEFAULT_ORDER, n=DEFAULT_N, start_value=0.0
) -> NDArray:
    """
    Cumulatively integrate the given discretely sampled signal in time using a Newton-Coases like formulation of
    requested order and layout. Note that higher order methods are only used in regions where the timestep is constant
    across the integration stencil- otherwise we fall back to the trapezoidal rule which can handle variable timesteps.
    A small amount of jitter (<1%) in timesteps is permitted though (and effectively ignored).

    NOTE: by default we start at 0.0 - which in general means that for a zero-mean process we will pick up a random
          offset that will need to be corracted afterwards. (out is not zero-mean).

    :param time: ndarray of length nt containing the elapsed time in seconds.
    :param signal: ndarray of length nt containing the signal to be integrated
    :param order: Order of the returned Newton-Coates integration approximation.
    :param n: number of future points in the integration stencil.
    :param start_value: Starting value of the integrated signal.
    :return: NDARRAY of length nt that contains the integrated signal that starts at the requested start_value.
    """

    primary_stencil = integration_stencil(order, n)
    primary_stencil_width = len(primary_stencil)

    integrated_signal = empty_like(signal)
    integrated_signal[0] = start_value
    integrated_signal[:] = 0.0

    number_of_constant_time_steps = 0
    prev_dt = time[1] - time[0]
    restart = True

    nt = len(signal)
    for ii in range(1, len(signal)):
        curr_dt = time[ii] - time[ii - 1]

        if ii + n - 1 < nt:
            future_dt = time[ii + n - 1] - time[ii + n - 2]
        else:
            future_dt = curr_dt
            restart = True

        if abs(future_dt - prev_dt) > 0.01 * curr_dt:
            # Jitter in the timestep, fall back to a lower order method that can handle this.
            restart = True
            number_of_constant_time_steps = 0

        if restart:
            number_of_constant_time_steps += 1
            stencil_width = 2
            stencil = array([0.5, 0.5])
            number_of_implicit_points = 1

            if number_of_constant_time_steps == primary_stencil_width:
                # We know have a series of enough constant timesteps to go to the higher order method.
                restart = False

        else:
            stencil_width = primary_stencil_width
            stencil = primary_stencil
            number_of_implicit_points = n

        delta = 0.0
        jstart = -(stencil_width - number_of_implicit_points)
        jend = number_of_implicit_points
        for jj in range(jstart, jend):
            delta += stencil[jj - jstart] * signal[ii + jj]

        integrated_signal[ii] = integrated_signal[ii - 1] + delta * curr_dt
        prev_dt = curr_dt

    return integrated_signal


@njit(cache=True)
def cumulative_distance(latitudes, longitudes):
    semi_major_axis = 6378137
    semi_minor_axis = 6356752.314245
    # eccentricity - squared
    eccentricity_squared = (
        semi_major_axis**2 - semi_minor_axis**2
    ) / semi_major_axis**2

    x = empty_like(latitudes)
    y = empty_like(longitudes)
    x[0] = 0
    y[0] = 0

    for ii in range(1, len(latitudes)):
        delta_longitude = (
            ((longitudes[ii] - longitudes[ii - 1] + 180) % 360 - 180) * pi / 180
        )
        delta_latitude = (latitudes[ii] - latitudes[ii - 1]) * pi / 180

        mean_latitude = (latitudes[ii] + latitudes[ii - 1]) / 2 * pi / 180

        # reduced latitude
        reduced_latitude = arctan(sqrt(1 - eccentricity_squared) * tan(mean_latitude))

        # length of a small meridian arc
        arc_length = (
            semi_major_axis
            * (1 - eccentricity_squared)
            * (1 - eccentricity_squared * sin(mean_latitude) ** 2) ** (-3 / 2)
        )

        x[ii] = x[ii - 1] + delta_longitude * semi_major_axis * cos(reduced_latitude)
        y[ii] = y[ii - 1] + arc_length * delta_latitude
    return x, y


@njit(cache=True)
def complex_response(
    normalized_frequency: NDArray, order: int, number_of_implicit_points: int = 1
):
    """
    The frequency complex response factor of the numerical integration scheme with given order and number of
    implicit points.

    :param normalized_frequency: Frequency normalized with the sampling frequency to calculate response factor at
    :param order: Order of the returned Newton-Coates integration approximation.
    :param number_of_implicit_points: number of future points in the integration stencil.
    :return: complex NDArray of same length as the input frequency containing the response factor at the given
             frequencies
    """
    number_of_explicit_points = order - number_of_implicit_points
    stencil = integration_stencil(order, number_of_implicit_points)

    normalized_omega = 2j * pi * normalized_frequency

    response_factor = zeros_like(normalized_frequency, dtype="complex_")
    for ii in range(-number_of_explicit_points, number_of_implicit_points):
        response_factor += stencil[ii + number_of_explicit_points] * exp(
            normalized_omega * ii
        )

    for index, omega in enumerate(normalized_omega):
        if omega == 0.0 + 0.0j:
            response_factor[index] = 1.0 + 0.0j
        else:
            response_factor[index] = response_factor[index] * (
                omega / (1 - exp(-omega))
            )

    return response_factor


@njit(cache=True)
def lagrange_base_polynomial_coef(order, base_polynomial_index):
    """
    We consider the interpolation of Y[0] Y[1] ... Y[order] spaced 1 apart at 0, 1,... point_index, ... order in terms
    of the Lagrange polynomial:

    Y[x]  =   L_0[x] Y[0] + L_1[x] Y[1] + .... L_order[x] Y[order].

    Here each of the lagrange polynomial coefficients L_n is expressed as a polynomial in x

    L_n = a_0 x**(order-1) + a_1 x**(order-2) + ... a_order

    where the coeficients may be found from the standard definition of the base polynomial (e.g. for L_0)

          ( x - x_1) * ... * (x - x_order )         ( x- 1) * (x-2) * ... * (x - order)
    L_0 = ------------------------------------  =  -------------------------------------
          (x_0 -x_1) * .... * (x_0 - x_order)        -1 * -2 * .... * -order

    where the right hand side follows after substituting x_n = n (i.e. 1 spacing). This function returns the set of
    coefficients [ a_0, a_1, ..., a_order ].

    :param order: order of the base polynomials.
    :param base_polynomial_index: which of the base polynomials to calculate
    :return: set of polynomial coefficients [ a_0, a_1, ..., a_order ]
    """
    poly = zeros(order + 1)
    poly[0] = 1
    denominator = 1
    jj = 0
    for ii in range(0, order + 1):
        if ii == base_polynomial_index:
            continue
        jj += 1

        # calculate the denomitor by repeated multiplication
        denominator = denominator * (base_polynomial_index - ii)

        # Calculate the polynomial coeficients. We start with a constant function (a_0=1) of order 0 and multiply this
        # polynomial with the next term ( x - x_0), (x-x_1) etc.
        poly[1 : jj + 1] += -ii * poly[0:jj]

    return poly / denominator


@njit(cache=True)
def integrated_lagrange_base_polynomial_coef(order, base_polynomial_index):
    """
    Calculate the polynomial coefficents of the integrated base polynomial.

    :param order: polynomial order of the interated base_polynomial.
    :param base_polynomial_index: which of the base polynomials to calculate
    :return: set of polynomial coefficients [ a_0, a_1, ..., a_[order-1], 0 ]
    """
    poly = zeros(order + 1)
    poly[0:order] = lagrange_base_polynomial_coef(order - 1, base_polynomial_index)

    # Calculate the coefficients of the integrated polynimial.
    for ii in range(order - 1):
        poly[ii] = poly[ii] / (order - ii)
    return poly


@njit(cache=True)
def evaluate_polynomial(poly, x):
    """
    Eval a polynomial at location x.
    :param poly: polynomial coeficients [a_0, a_1, ..., a_[order+1]]
    :param x: location to evaluate the polynomial/
    :return: value of the polynomial at the location
    """
    res = 0
    order = len(poly) - 1
    for ii in range(0, order + 1):
        res += poly[ii] * x ** (order - ii)
    return res


@njit(cache=True)
def integration_stencil(order: int, number_of_implicit_points: int = 1) -> NDArray:
    """
    Find the Newton-Coastes like- integration stencil given the desired order and the number of "implicit" points.
    Specicially, let the position z at instance t[j-1] be known, and we wish to approximate z at time t[j], where
    t[j] - t[j-1] = dt  for all j, given the velocities w[j]. This implies we solve

            dz
           ---- = w    ->    z[j] = z[j-1] + dz     with dz = Integral[w] ~ dt * F[w]
            dt

    To solve the integral we use Newton-Coates like approximation and express w(t) as a function of points w[j+i],
    where i = -m-1 ... n-1 using a Lagrange Polynomial. Specifically we consider points in the past and future as we
    anticipate we can buffer w values in any application.

    In this framework the interval of interest lies between j-1, and j  (i=0 and 1).

        j-m-1  ...  j-2  j-1   j   j+1  ...  j+n-1
          |    |    |    |----|    |    |    |

    The number of points used will be refered to ast the order = n+m+1. The number of points with i>=0 will be referred to as
    the number of implicit points, so that n = number_of_implicit_points. The number of points i<0 is the number of
    explicit points m = order - n - 1.

    This function calculates the weights such that

    dz  =    weights[0] w[j-m] + ... +  weights[m-1] w[j-1] + weights[m] w[j] + ... weights[order-1] w[j+n-1]

    :param order: Order of the returned Newton-Coates set of coefficients.
    :param number_of_implicit_points: number of points for which i>0
    :return: Numpy array of length Order with the weights.
    """
    weights = zeros(order)
    number_of_explicit_points = order - number_of_implicit_points

    for ii in range(0, order):
        # Get the polynomial coefficients assocated with the ii'th lagrangian base polynomial l_ii
        base_poly = integrated_lagrange_base_polynomial_coef(order, ii)

        # Calculate the dz from the evaluation of the indeterminate integrals
        weights[ii] = evaluate_polynomial(
            base_poly, number_of_explicit_points
        ) - evaluate_polynomial(base_poly, number_of_explicit_points - 1)

    return weights
