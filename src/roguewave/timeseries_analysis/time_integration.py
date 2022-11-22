from numba import njit
from numpy import empty_like, arctan, tan, sqrt, cos, sin, pi
from numpy.typing import NDArray


@njit(cache=True)
def integrate(time: NDArray, signal: NDArray) -> NDArray:
    coef = [3 / 8, 19 / 24, -5 / 24, 1 / 24]

    integrated_signal = empty_like(signal)
    integrated_signal[0] = 0

    # Start with Trapezoidal rule
    for ii in range(1, 3):
        dt = time[ii] - time[ii - 1]
        integrated_signal[ii] = (
            integrated_signal[ii - 1] + (signal[ii] + signal[ii - 1]) / 2 * dt
        )

    # Then apply Adams Moulton
    for ii in range(3, len(signal)):
        dt = time[ii] - time[ii - 1]
        if dt > 0.41:
            integrated_signal[ii] = 0.0
            continue

        integrated_signal[ii] = (
            integrated_signal[ii - 1]
            + coef[0] * signal[ii] * dt
            + coef[1] * signal[ii - 1] * dt
            + coef[2] * signal[ii - 2] * dt
            + coef[3] * signal[ii - 3] * dt
        )
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
