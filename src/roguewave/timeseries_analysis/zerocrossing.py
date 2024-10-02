import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def lower_index_zero_interval(signal):
    """
    Define waves as the zero crossings of the signal.
    """
    return np.where(np.diff(np.sign(signal))<0)[0]

def wave_mask(signal):
    """
    Define waves as the zero crossings of the signal.
    """
    return np.cumsum(np.diff(np.sign(signal))<0)

def time_zero(time,signal):
    lower_index = lower_index_zero_interval(signal)

    if lower_index[-1] == len(signal):
        lower_index = lower_index[:-1]

    upper_index = lower_index + 1

    delta_signal = -signal[lower_index]/(signal[upper_index]-signal[lower_index])
    delta_time = time[upper_index]-time[lower_index]

    zero_crossing_time = time[lower_index] + signal[lower_index] * delta_time/ delta_signal
    return zero_crossing_time

@njit(cache=True)
def zero_crossing_extrema(signal):
    """
    Find the extrema of a signal.
    """
    indices = lower_index_zero_interval(signal)

    number_of_waves = len(indices) - 1
    maxima = np.zeros(number_of_waves) - np.inf
    minima = np.zeros(number_of_waves) + np.inf

    for wave_index in range(0,number_of_waves):
        lower_index = indices[wave_index]
        upper_index = indices[wave_index+1]

        for signal_index in range(lower_index,upper_index):
            if signal[signal_index] > maxima[wave_index]:
                maxima[wave_index] = signal[signal_index]
            if signal[signal_index] < minima[wave_index]:
                minima[wave_index] = signal[signal_index]

    return maxima, minima

def zero_crossing_wave_heights(signal):
    maxima, minima = zero_crossing_extrema(signal)
    return maxima - minima

def zero_crossing_period(time, signal):
    zero_crossing_time = time_zero(time,signal)
    return np.diff(zero_crossing_time)

def maximum_wave_height(signal):
    _waveheights = zero_crossing_wave_heights(signal)
    return np.max(_waveheights)

