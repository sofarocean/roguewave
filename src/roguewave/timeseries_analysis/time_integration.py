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


@njit(cache=True)
def integrate(
    time: NDArray, signal: NDArray, order=4, primary_number_of_implicit_points=1
) -> NDArray:

    primary_stencil = integration_stencil(order, primary_number_of_implicit_points)
    primary_stencil_width = len(primary_stencil)

    integrated_signal = empty_like(signal)
    integrated_signal[0] = 0
    integrated_signal[:] = 0.0

    number_of_constant_time_steps = 0
    prev_dt = time[1] - time[0]
    restart = True

    nt = len(signal)
    for ii in range(1, len(signal)):
        curr_dt = time[ii] - time[ii - 1]

        if ii + primary_number_of_implicit_points - 1 < nt:
            future_dt = (
                time[ii + primary_number_of_implicit_points - 1]
                - time[ii + primary_number_of_implicit_points - 2]
            )
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
            number_of_implicit_points = primary_number_of_implicit_points

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
def complex_response(normalized_frequency, order, number_of_implicit_points=1):
    number_of_explicit_points = order - number_of_implicit_points
    stencil = integration_stencil(order, number_of_implicit_points)

    normalized_omega = 1j * 2 * pi * normalized_frequency
    response_factor = zeros_like(normalized_frequency, dtype="complex_")

    jstart = -number_of_explicit_points
    jend = number_of_implicit_points
    for ii in range(jstart, jend):
        response_factor += stencil[ii - jstart] * exp(normalized_omega * ii)

    for index, omega in enumerate(normalized_omega):
        if omega == 0.0 + 0.0j:
            response_factor[index] = 1.0 + 0.0j
        else:
            response_factor[index] = response_factor[index] * (
                omega / ((1 - exp(-omega)))
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
def integration_stencil(order, number_of_implicit_points=1):

    weights = zeros(order)
    istart = order - number_of_implicit_points - 1

    for ii in range(0, order):
        base_poly = integrated_lagrange_base_polynomial_coef(order, ii)
        weights[ii] = evaluate_polynomial(base_poly, istart + 1) - evaluate_polynomial(
            base_poly, istart
        )

    return weights
