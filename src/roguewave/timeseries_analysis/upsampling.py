from numpy.fft import rfft, irfft
from numpy import linspace


def upsample(signal, factor: int, t0=0, sampling_frequency=2.5):
    """
    Spectral upsampling. There will be edge effects unless the signal has been windowed.

    :param signal:
    :param factor:
    :return:
    """
    n = len(signal)
    upsampled_time = linspace(0, n * factor, n * factor, endpoint=False) / (
        sampling_frequency * factor
    )
    return upsampled_time, irfft(rfft(signal) * factor, n * factor)
