from numba import njit
from numba.typed import Dict
from numba.core import types
from numpy import empty_like, abs, nan, isfinite, flip, linspace
from numpy.typing import NDArray
from typing import Literal
from numpy import interp
from scipy.signal import sosfilt, sosfiltfilt, butter


@njit(cache=True)
def exponential_filter(
    time_seconds, signal, sampling_frequency=2.5, options: dict = None
):
    """
    Exponential filter that operates on the differences between succesive values.

    :param time_seconds:
    :param signal:
    :param options:
    :return:
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("smoothing_factor", 0.004)
    maximum_gap_size = options.get("maximum_gap_size", 3)
    sampling_interval = 1 / sampling_frequency

    # Initialize empty array
    filtered_signal = empty_like(signal)

    # Initialize start values
    exponential_mean = 0
    filtered_signal[0] = 0

    for ii in range(1, len(signal)):
        if (
            time_seconds[ii] - time_seconds[ii - 1]
            > maximum_gap_size * sampling_interval
        ):
            # Restart
            exponential_mean = 0.0
            filtered_signal[ii] = 0.0

        exponential_mean = (
            exponential_mean * (1 - smoothing_factor) + signal[ii] * smoothing_factor
        )
        filtered_signal[ii] = signal[ii] - exponential_mean

    return filtered_signal


@njit(cache=True)
def exponential_delta_filter(
    time_seconds, signal, sampling_frequency=2.5, options: dict = None
):
    """
    Exponential filter that operates on the differences between succesive values.

    :param time_seconds:
    :param signal:
    :param options:
    :return:
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("smoothing_factor", 0.004)
    maximum_gap_size = options.get("maximum_gap_size", 3)
    sampling_interval = 1 / sampling_frequency

    # Initialize empty array
    filtered_signal = empty_like(signal)

    # Initialize start values
    exponential_mean = 0
    filtered_signal[0] = 0

    for ii in range(1, len(signal)):
        signal_delta = signal[ii] - signal[ii - 1]

        if (
            time_seconds[ii] - time_seconds[ii - 1]
            > maximum_gap_size * sampling_interval
        ):
            # Restart
            exponential_mean = 0.0
            filtered_signal[ii] = 0.0

        exponential_mean = (
            exponential_mean * (1 - smoothing_factor) + signal_delta * smoothing_factor
        )
        filtered_signal[ii] = filtered_signal[ii - 1] + (
            signal_delta - exponential_mean
        )

    return filtered_signal


@njit(cache=True)
def spike_filter(time: NDArray, signal: NDArray, options: dict = None) -> NDArray:
    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    rejection_threshold = options.get("rejection_threshold", 9.81)

    filtered_signal = empty_like(signal)
    filtered_signal[0] = signal[0]

    prev_signal = signal[0]
    prev_time = time[0]

    for ii in range(1, len(signal)):
        delta_signal = signal[ii] - prev_signal
        delta_time = time[ii] - prev_time

        if abs(delta_signal / delta_time) > rejection_threshold:
            filtered_signal[ii] = nan
            continue

        filtered_signal[ii] = prev_signal + delta_signal
        prev_signal = filtered_signal[ii]
        prev_time = time[ii]
    return nan_interpolate(time, filtered_signal)


@njit(cache=True)
def cumulative_filter(time: NDArray, signal: NDArray, options: dict = None) -> NDArray:
    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("smoothing_factor", 0.01)

    filtered_signal = empty_like(signal)
    filtered_signal[0] = signal[0]
    cumchange = 0.0

    lf = 0.0
    istart = 0
    for ii in range(1, len(signal)):
        delta_signal = signal[ii] - signal[ii - 1]

        lf = lf * (1 - smoothing_factor) + delta_signal * smoothing_factor
        delta = delta_signal - lf

        if cumchange * (cumchange + delta) < 0:
            istart = ii
        else:
            if ii - istart > 23:
                filtered_signal[istart:ii] = filtered_signal[
                    istart:ii
                ] - cumchange * linspace(0, 1, ii - istart)
                istart = ii
                cumchange = 0.0

        cumchange += delta
        filtered_signal[ii] = filtered_signal[ii - 1] + delta

    return nan_interpolate(time, filtered_signal)


def sos_filter(
    signal: NDArray, direction: Literal["backward", "forward", "filtfilt"], sos=None
) -> NDArray:
    #
    # Apply forward/backward/filtfilt sos filter
    #
    # Get SOS coefficients
    if sos is None:
        sos = butter(4, 0.033, btype="high", output="sos", fs=2.5)
    #
    if direction == "backward":
        return flip(sosfilt(sos, flip(signal)))

    elif direction == "forward":
        return sosfilt(sos, signal)

    elif direction == "filtfilt":
        return sosfiltfilt(sos, signal)

    else:
        raise ValueError(f"Unknown direction {direction}")


@njit(cache=True)
def nan_interpolate(time: NDArray, signal: NDArray) -> NDArray:
    mask = isfinite(signal)
    return interp(time, time[mask], signal[mask])
