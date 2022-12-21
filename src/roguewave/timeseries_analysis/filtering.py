from numba import njit
from numba.typed import Dict
from numba.core import types
from numpy import empty_like, abs, nan, isfinite, flip, linspace, sqrt
from numpy.typing import NDArray
from typing import Literal
from numpy import interp
from scipy.signal import sosfilt, sosfiltfilt, butter


@njit(cache=True)
def exponential_filter(time_seconds, signal, options: dict = None):
    """
    Exponential filter that operates on the differences between successive values.

    :param time_seconds:
    :param signal:
    :param options:
    :return:
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("smoothing_factor", 0.004)
    maximum_gap_size = options.get("maximum_gap_size", 3)
    sampling_frequency = options.get("sampling_frequency", 2.5)
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
        ) or (not isfinite(signal[ii])):
            # Restart
            exponential_mean = 0.0
            filtered_signal[ii] = 0.0
            continue

        exponential_mean = (
            exponential_mean * (1 - smoothing_factor) + signal[ii] * smoothing_factor
        )
        filtered_signal[ii] = signal[ii] - exponential_mean

    return filtered_signal


@njit(cache=True)
def exponential_delta_filter(time_seconds, signal, options: dict = None):
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
    sampling_frequency = options.get("sampling_frequency", 2.5)
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
        ) or (not isfinite(signal_delta)):
            # Restart
            exponential_mean = 0.0
            filtered_signal[ii] = 0.0
            continue

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
def cumulative_filter(signal: NDArray, options: dict = None) -> NDArray:
    """
    Filter an input signal to remove step-like changes according to a cumulative filter approach.

    :param signal: Input signal at a constant sampling interval. The signal and its first order difference are
    observations of zero-mean processes.
    :param options: optional dictionary to set algorithm parameters.
    :return: Filtered signal with step like changes to the mean removed.
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("cumulative_smoothing_factor", 0.01)
    scale_factor = options.get("scale_factor", 5.0)
    correct = options.get("correct", 1.0)

    # Allocate the output filtered signal and set the first value to the first entry in the signal.
    filtered_signal = empty_like(signal)
    filtered_signal[0] = signal[0]

    # Initialize algorithm paramteres
    cumulative_distance = (
        0.0  # Cumulative change in the function since the last delta-zero-crossing
    )
    prev_velocity = 0.0  # previous difference
    idx_last_zero = 0  # Index of the last sign change
    variance = 4  # Variance in the cumulative difference.
    change_in_mean_detected = False  # Did we undergo a likely stepchange in the current upward or downward drift of the
    # signal?

    for current_idx in range(1, len(signal)):
        velocity = signal[current_idx] - signal[current_idx - 1]

        if prev_velocity * velocity < 0:
            # flip in sign of the differences indicates the "velocity" changed sign. Or a zero-crossing of the velocity/
            # difference signal.

            if not change_in_mean_detected:
                # No step change. Update the mean of the variance.
                variance = (
                    variance * (1 - smoothing_factor)
                    + cumulative_distance**2 * smoothing_factor
                )

            else:
                # Step change. Remove the linear trend from the current position to the last known zero-crossing.
                if correct == 1.0:
                    filtered_signal[idx_last_zero:current_idx] = filtered_signal[
                        idx_last_zero:current_idx
                    ] - cumulative_distance * linspace(
                        0, 1, current_idx - idx_last_zero
                    )
                else:
                    filtered_signal[idx_last_zero + 1 : current_idx - 1] = nan
                    filtered_signal[current_idx - 1] = filtered_signal[idx_last_zero]

            idx_last_zero = current_idx
            cumulative_distance = 0.0
            change_in_mean_detected = False
        else:
            #
            # Check how far have we moved up (or down) since the last channge in sign of the velocity signal. If the
            # cumulative change is larger than a typical change we assume that there has been a change in the mean. Note
            # That no corrective action is taken here. We only correct the signal once velocities have flipped sign.
            if abs(cumulative_distance) > scale_factor * sqrt(variance):
                change_in_mean_detected = True

        # Update previous velocity, cumulative signal and the filtered signal.
        prev_velocity = velocity
        cumulative_distance += velocity
        filtered_signal[current_idx] = filtered_signal[current_idx - 1] + velocity

    return filtered_signal


@njit(cache=True)
def cumsum(signal: NDArray, options: dict = None) -> NDArray:
    """
    Filter an input signal to remove step-like changes according to a cumulative filter approach.

    :param signal: Input signal at a constant sampling interval. The signal and its first order difference are
    observations of zero-mean processes.
    :param options: optional dictionary to set algorithm parameters.
    :return: Filtered signal with step like changes to the mean removed.
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    smoothing_factor = options.get("cumulative_smoothing_factor", 0.004)
    # scale_factor = options.get("scale_factor", 5.0)
    # correct = options.get("correct", 0.0)

    # Allocate the output filtered signal and set the first value to the first entry in the signal.
    filtered_signal = empty_like(signal)
    filtered_signal[0] = signal[0]

    # Initialize algorithm paramteres
    cumsum_h = 0.0
    cumsum_l = 0.0

    idx_last_zero = 0  # Index of the last sign change

    signal_variance = 0.0
    signal_mean = 0.0
    drift = 0.0

    prev_drift = 0
    was_unstable = False
    last_stable_drift = 0
    for current_idx in range(1, len(signal)):
        velocity = signal[current_idx] - signal[current_idx - 1]

        signal_mean = signal_mean * (1 - smoothing_factor) + velocity * smoothing_factor
        signal_variance = (
            signal_variance * (1 - smoothing_factor)
            + (signal[current_idx] - signal_mean) ** 2
        )

        cumsum_h = cumsum_h + (
            velocity - signal_mean
        )  # - 0.000001  #(velocity - signal_mean)
        cumsum_l = cumsum_l - (
            velocity - signal_mean
        )  # - 0.000001   #(velocity - signal_mean)

        if cumsum_h < 0.0:
            drift += cumsum_h
            cumsum_h = 0.0

        if cumsum_l < 0.0:
            drift -= cumsum_l
            cumsum_l = 0.0

        if drift == prev_drift:

            if was_unstable:
                if abs(drift - last_stable_drift) > 2:
                    filtered_signal[idx_last_zero + 1 : current_idx - 1] = -10
                    filtered_signal[current_idx - 1] = filtered_signal[idx_last_zero]

            idx_last_zero = current_idx
            was_unstable = False
            last_stable_drift = drift
        else:
            was_unstable = True

        prev_drift = drift
        filtered_signal[current_idx] = filtered_signal[current_idx - 1] + velocity

    return filtered_signal


def sos_filter(
    signal: NDArray, direction: Literal["backward", "forward", "filtfilt"], **kwargs
) -> NDArray:
    #
    # Apply forward/backward/filtfilt sos filter
    #
    # Get SOS coefficients

    mask = ~isfinite(signal)
    output = signal.copy()
    output = nan_interpolate(
        linspace(0, len(signal), len(signal), endpoint=False), output
    )

    sos = kwargs.get("sos", None)
    if sos is None:
        sos = butter(4, 0.033, btype="high", output="sos", fs=2.5)

    if direction == "backward":
        output = flip(sosfilt(sos, flip(output)))

    elif direction == "forward":
        output = sosfilt(sos, output)

    elif direction == "filtfilt":
        output = sosfiltfilt(sos, output)

    else:
        raise ValueError(f"Unknown direction {direction}")
    output[mask] = nan
    return output


@njit(cache=True)
def nan_interpolate(time: NDArray, signal: NDArray) -> NDArray:
    mask = isfinite(signal)
    return interp(time, time[mask], signal[mask])
