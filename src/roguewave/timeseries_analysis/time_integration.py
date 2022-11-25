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
    flip,
)
from numpy.typing import NDArray


@njit(cache=True)
def integration_coeficients(method="adams_moulton_4th_order"):
    if method == "adams_moulton_4th_order":
        return (4, array((3 / 8, 19 / 24, -5 / 24, 1 / 24)))

    elif method == "adams_moulton_3th_order":
        return (3, array((5 / 12, 2 / 3, -1 / 12)))

    elif method == "pieter":
        return (3, array((-1.75, 2.5, 0.25)))

    elif method == "adams_moulton_5th_order":
        return (5, array((251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720)))

    elif method == "adams_moulton_6th_order":
        return (6, array((475, 1427, -798, 482, -173, 27)) / 1440.0)

    elif method == "adams_moulton_6th_order":
        return (6, array((475, 1427, -798, 482, -173, 27)) / 1440.0)

    elif method == "adams_moulton_7th_order":
        return (7, array((19087, 65112, -46461, 37504, -20211, 6312, -863)) / 60480.0)

    elif method == "adams_moulton_8th_order":
        return (
            8,
            array((36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375))
            / 120960.0,
        )

    elif method == "trapezoidal":
        return (2, array((0.5, 0.5)))

    else:
        return (2, array((0.5, 0.5)))


@njit(cache=True)
def integrate(
    time: NDArray, signal: NDArray, method="adams_moulton_4th_order"
) -> NDArray:
    primary_stencil_width, primary_stencil = integration_coeficients(method=method)
    integrated_signal = empty_like(signal)
    integrated_signal[0] = 0

    number_of_constant_time_steps = 0
    prev_dt = time[1] - time[0]
    restart = True
    for ii in range(0, len(signal)):
        curr_dt = time[ii] - time[ii - 1]

        if abs(curr_dt - prev_dt) > 0.01 * curr_dt:
            # Jitter in the timestep, fall back to a lower order method that can handle this.
            restart = True
            number_of_constant_time_steps = 0

        if restart:
            number_of_constant_time_steps += 1
            stencil_width = 2
            stencil = (0.5, 0.5)

            if number_of_constant_time_steps == primary_stencil_width:
                # We know have a series of enough constant timesteps to go to the higher order method.
                restart = False

        else:
            stencil_width = primary_stencil_width
            stencil = primary_stencil

        delta = 0.0
        for jj in range(0, stencil_width):
            delta += stencil[jj] * signal[ii - jj]

        integrated_signal[ii] = integrated_signal[ii - 1] + delta * curr_dt
        prev_dt = curr_dt

    return integrated_signal


@njit(cache=True)
def integratev2(
    time: NDArray, signal: NDArray, method="adams_moulton_4th_order"
) -> NDArray:
    primary_stencil_width, primary_stencil = integration_coeficients(method=method)
    integrated_signal = empty_like(signal)
    integrated_signal[0] = 0

    number_of_constant_time_steps = 0
    prev_dt = time[1] - time[0]
    restart = True
    for ii in range(0, len(signal)):
        curr_dt = time[ii] - time[ii - 1]

        if abs(curr_dt - prev_dt) > 0.01 * curr_dt:
            # Jitter in the timestep, fall back to a lower order method that can handle this.
            restart = True
            number_of_constant_time_steps = 0

        if restart:
            number_of_constant_time_steps += 1
            stencil_width = 2
            stencil = (0.5, 0.5)

            if number_of_constant_time_steps == primary_stencil_width:
                # We know have a series of enough constant timesteps to go to the higher order method.
                restart = False

        else:
            stencil_width = primary_stencil_width
            stencil = primary_stencil

        delta = 0.0
        for jj in range(0, stencil_width):
            delta += stencil[jj] * signal[ii - jj]

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
def complex_response(normalized_frequency, method="adams_moulton_4th_order"):
    stencil_width, stencil = integration_coeficients(method=method)

    normalized_omega = 1j * 2 * pi * normalized_frequency
    response_factor = zeros_like(normalized_frequency, dtype="complex_")

    for ii in range(0, stencil_width):
        response_factor += stencil[ii] * exp(-normalized_omega * ii)

    for index, omega in enumerate(normalized_omega):
        if omega == 0.0 + 0.0j:
            response_factor[index] = 1.0 + 0.0j
        else:
            response_factor[index] = response_factor[index] * (
                omega / ((1 - exp(-omega)))
            )

    return response_factor


@njit(cache=True)
def complex_responsev2(normalized_frequency, order, number_of_implicit_points=1):
    number_of_explicit_points = order - number_of_implicit_points
    stencil = get_stencil(order, number_of_implicit_points)

    normalized_omega = 1j * 2 * pi * normalized_frequency
    response_factor = zeros_like(normalized_frequency, dtype="complex_")

    for ii in range(-number_of_implicit_points + 1, number_of_explicit_points + 1):
        response_factor += stencil[ii + number_of_implicit_points - 1] * exp(
            -normalized_omega * ii
        )

    for index, omega in enumerate(normalized_omega):
        if omega == 0.0 + 0.0j:
            response_factor[index] = 1.0 + 0.0j
        else:
            response_factor[index] = response_factor[index] * (
                omega / ((1 - exp(-omega)))
            )

    return response_factor


@njit(cache=True)
def lagrange_poly(order, which):
    poly = zeros(order)
    poly[0] = 1

    denominator = 1
    jj = 0
    for ii in range(0, order):
        if ii == which:
            continue
        jj += 1
        denominator = denominator * (which - ii)
        poly[1 : jj + 1] += -ii * poly[0:jj]

    return poly / denominator


@njit(cache=True)
def integrated_lagrange_poly(order, which):
    poly = zeros(order + 1)
    poly[0:order] = lagrange_poly(order, which)

    for ii in range(order - 1):
        poly[ii] = poly[ii] / (order - ii)
    return poly


@njit(cache=True)
def eval_poly(poly, x):
    res = 0
    order = len(poly) - 1
    for ii in range(0, order + 1):
        res += poly[ii] * x ** (order - ii)
    return res


@njit(cache=True)
def integrated_lagrange_coef(order, which, x0, x1):
    poly = integrated_lagrange_poly(order, which)
    return eval_poly(poly, x1) - eval_poly(poly, x0)


@njit(cache=True)
def get_lagrange_weights(order, istart=None):
    weights = zeros(order)
    if istart is None:
        istart = order - 2

    for ii in range(0, order):
        weights[ii] = integrated_lagrange_coef(order, ii, istart, istart + 1)
    return flip(weights)


@njit(cache=True)
def get_stencil(order, number_of_implicit_points=1):
    number_of_explicit_points = order - number_of_implicit_points
    return get_lagrange_weights(order, number_of_explicit_points - 1)


if __name__ == "__main__":
    stencil = get_stencil(3, 1)
    print(array((3 / 8, 19 / 24, -5 / 24, 1 / 24)))
    print(stencil)
