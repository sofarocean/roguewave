from numba import njit
from numpy import ndarray, empty_like, diff, abs, nan, isfinite
from numpy.typing import NDArray
from numpy import interp
from roguewave.timeseries_analysis.parse_spotter_files import (
    load_gps,
    load_displacement,
)


def pipeline(filters, time: ndarray, signal: ndarray) -> ndarray:

    for (filter, settings) in filters:
        signal = filter(time, signal, **settings)

    return signal


@njit(cache=True)
def exponential_highpass_filter(time_seconds, signal):
    sampling_frequency = 2.5  # kwargs.get('sampling_frequency', 2.5)
    sampling_interval = 1 / sampling_frequency
    smoothing_factor = 0.05  # kwargs.get('smoothing_factor',0.01)

    exponential_mean = 0
    filtered_signal = empty_like(signal)
    for ii in range(len(signal)):
        if ii > 0:
            time_delta_seconds = time_seconds[ii] - time_seconds[ii - 1]
        else:
            time_delta_seconds = sampling_interval

        if abs(time_delta_seconds - sampling_interval) > 0.1 * sampling_interval:
            # restart the filter
            exponential_mean = 0

        exponential_mean = (
            exponential_mean * (1 - smoothing_factor) + signal[ii] * smoothing_factor
        )
        filtered_signal[ii] = signal[ii] - exponential_mean

    return filtered_signal


@njit(cache=True)
def outlier_rejection_filter(time: NDArray, signal: NDArray) -> NDArray:
    sampling_frequency = 2.5
    rejection_threshold = 9.81 / 2  # 9.81 * 0.45
    sampling_interval = 1 / sampling_frequency

    filtered_signal = empty_like(signal)
    filtered_signal[0] = signal[0]

    prev_signal = signal[0]
    prev_time = time[0]

    for ii in range(1, len(signal)):
        delta_signal = signal[ii] - prev_signal
        delta_time = time[ii] - time[ii - 1]

        if abs(delta_signal / delta_time) > rejection_threshold:
            filtered_signal[ii] = nan
            continue

        filtered_signal[ii] = 0.1 * (prev_signal + delta_signal)  # + 0.3*signal[ii]
        prev_signal = filtered_signal[ii]
        prev_time = time[ii]
    return nan_interpolate(time, filtered_signal)


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
        integrated_signal[ii] = (
            integrated_signal[ii - 1]
            + coef[0] * signal[ii] * dt
            + coef[1] * signal[ii - 1] * dt
            + coef[2] * signal[ii - 2] * dt
            + coef[3] * signal[ii - 3] * dt
        )
    return integrated_signal


@njit(cache=True)
def nan_interpolate(time: NDArray, signal: NDArray) -> NDArray:
    mask = isfinite(signal)
    return interp(time, time[mask], signal[mask])


if __name__ == "__main__":
    vert = load_gps(path="/Users/pietersmit/Downloads/Sunflower13/log")
    hor = load_displacement(path="/Users/pietersmit/Downloads/Sunflower13/log")
