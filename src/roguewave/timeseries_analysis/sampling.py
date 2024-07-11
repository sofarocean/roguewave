from numpy.fft import rfft, irfft
import numpy as np
import scipy
from scipy.signal.windows import tukey

def sample_irregular_signal(time_seconds: np.ndarray, signal: np.ndarray, sampling_frequency: float):
    """
    Sample an irregular signal at a constant rate. We do this through interpolation.

    :param time_seconds: time in seconds
    :param signal: signal
    :param sampling_frequency: sampling frequency in Hz
    :return: interpolated time, interpolated signal
    """

    # Interpolate the time base to integer values
    time_base = (time_seconds - time_seconds[0]) * sampling_frequency
    interpolated_time_base = np.arange(np.floor(time_base[-1]) + 1)
    interpolator = scipy.interpolate.interp1d( time_base, signal,kind='quadratic',assume_sorted=True)
    interpolated_signal = interpolator(interpolated_time_base)
    return interpolated_time_base / sampling_frequency + time_seconds[0], interpolated_signal

def upsample(signal, factor: int, t0=0, sampling_frequency=2.5):
    """
    Spectral upsampling. There will be edge effects unless the signal has been windowed.

    :param signal:
    :param factor:
    :return:
    """
    n = len(signal)
    upsampled_time = (
        np.linspace(0, n * factor, n * factor, endpoint=False)
        / (sampling_frequency * factor)
        + t0
    )

    return upsampled_time, irfft(rfft(signal) * factor, n * factor)

def resample( sample_time, time, signal):

    desired_sampling_frequency = 1/np.diff(sample_time).min()

    sampling_frequency = 1/np.diff(time).mean()
    relative_sampling_time = time * sampling_frequency % 1 - 0.5
    if np.abs(relative_sampling_time).max() > 0.01:
        regular_time, regular_sampled_signal = sample_irregular_signal(time, signal, sampling_frequency)
    else:
        regular_time = time
        regular_sampled_signal = signal

    relative_difference = desired_sampling_frequency / sampling_frequency

    upsample_factor = max( 5*int(np.ceil(relative_difference)), 5)
    upsampled_time, upsampled_signal_spectral \
        = upsample( regular_sampled_signal, upsample_factor, regular_time[0], sampling_frequency )

    upsampled_signal_spline = scipy.interpolate.interp1d( regular_time, regular_sampled_signal,
                                                          kind='quadratic', assume_sorted=True)(upsampled_time)

    delta = upsampled_signal_spectral - upsampled_signal_spline

    upsampled_signal = upsampled_signal_spline + tukey( len(upsampled_signal_spline), 0.01) * delta
    return np.interp( sample_time, upsampled_time, upsampled_signal, left=np.nan, right=np.nan)
