# Author: Pieter Bart Smit

"""
A set of spectral analysis routines specifically designed to work with data
from Sofar Spotter ocean wave buoys.

Classes:

- `SpectralAnalysisConfig`, configuration object

Functions:

- `segment_timeseries`, split timeseries into non-overlapping segments
- `calculate_moments`, calculate directional moments from (co)variances
- `spectral_analysis`, calculate wave spectrum for specific segment
- `generate_spectra`, calculate wave spectra for all segments
- `window_power_correction_factor`, correct for power loss due to windowing

How To Use This Module
======================
(See the individual classes, methods, and attributes for details.)

1. Import it: ``import welch`` or ``from welch import ...``.
s
"""

import numpy
from numpy.fft import fft
from typing import List
from roguewave import FrequencySpectrum
from numpy.typing import NDArray
from typing import Tuple
from scipy.signal.windows import get_window
from xarray import Dataset
from numba import njit, objmode


@njit(cache=True)
def segment_timeseries(epoch_time, segment_length_seconds) -> List[tuple]:
    """
    This function segments the time series into segments of the given length.
    It returns a list of tuples containing the indices of the start and end
    point of each segment. Segmens are open ended [t_start,t_end>.

    If the segment length does not fit a whole number of times into the vector
    the partial remaining segment is discarded.

    :param epoch_time: time vector (seconds, unix epoch)
    :param segment_length_seconds: lenght of segment in segments
    :return:
    """

    # Initialize the start time and the expected next segment start
    istart = 0
    t_start = epoch_time[0]
    t_next = t_start + segment_length_seconds

    segments = []
    # loop over all times
    for index, time in enumerate(epoch_time):
        # if the current time is larger or equal to the next segment start,
        # create a new segment
        if time > t_next:
            # Add segment to list
            segments.append((istart, index))

            # setup the new start time/index and next expected segment start
            istart = index
            t_start = t_next
            t_next = t_start + segment_length_seconds

    return segments


@njit(cache=True)
def power_spectra(
    x: numpy.ndarray,
    y: numpy.ndarray,
    z: numpy.ndarray,
    window: numpy.ndarray,
    sampling_frequency,
    window_overlap_fraction=0.5,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    overlap = int(len(window) * window_overlap_fraction)
    nyquist_index = len(window) // 2

    sxx = numpy.zeros((nyquist_index), dtype="complex_")
    syy = numpy.zeros((nyquist_index), dtype="complex_")
    szz = numpy.zeros((nyquist_index), dtype="complex_")
    sxy = numpy.zeros((nyquist_index), dtype="complex_")
    szx = numpy.zeros((nyquist_index), dtype="complex_")
    szy = numpy.zeros((nyquist_index), dtype="complex_")

    x = x - numpy.mean(x)
    y = y - numpy.mean(y)
    z = z - numpy.mean(z)

    ii = 0
    istart = 0
    while True:
        # Advance the counter

        iend = istart + len(window)
        if iend > len(x):
            break

        ii += 1

        with objmode(
            fft_x="complex128[:]", fft_y="complex128[:]", fft_z="complex128[:]"
        ):
            # FFT not supported in Numba- yet
            fft_x = fft(x[istart:iend] * window)[0:nyquist_index]
            fft_y = fft(y[istart:iend] * window)[0:nyquist_index]
            fft_z = fft(z[istart:iend] * window)[0:nyquist_index]

        szz += 2 * fft_z * numpy.conjugate(fft_z)
        syy += 2 * fft_y * numpy.conjugate(fft_y)
        sxx += 2 * fft_x * numpy.conjugate(fft_x)
        szx += 2 * fft_z * numpy.conjugate(fft_x)
        szy += 2 * fft_z * numpy.conjugate(fft_y)
        sxy += 2 * fft_x * numpy.conjugate(fft_y)

        istart = istart + len(window) - overlap

    frequency_step = sampling_frequency / len(window)
    factor = (
        window_power_correction_factor(window) ** 2
        / ii
        / frequency_step
        / len(window) ** 2
    )
    szz = numpy.real(szz) * factor
    syy = numpy.real(syy) * factor
    sxx = numpy.real(sxx) * factor
    cxy = numpy.real(sxy) * factor
    qzx = numpy.imag(szx) * factor
    qzy = numpy.imag(szy) * factor

    a1 = qzx / numpy.sqrt(szz * (sxx + syy))
    b1 = qzy / numpy.sqrt(szz * (sxx + syy))
    a2 = (sxx - syy) / (sxx + syy)
    b2 = 2 * cxy / (sxx + syy)

    return szz, a1, b1, a2, b2


@njit(cache=True)
def welch(
    epoch_time: numpy.ndarray,
    x: numpy.ndarray,
    y: numpy.ndarray,
    z: numpy.ndarray,
    window: numpy.ndarray,
    sampling_frequency: float,
    segment_length_seconds=1800,
    window_overlap_fraction=0.5,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    assert len(x) == len(y) == len(z) == len(epoch_time)

    segments = segment_timeseries(epoch_time, segment_length_seconds)

    e = numpy.empty((len(segments), len(window) // 2))
    a1 = numpy.empty((len(segments), len(window) // 2))
    b1 = numpy.empty((len(segments), len(window) // 2))
    a2 = numpy.empty((len(segments), len(window) // 2))
    b2 = numpy.empty((len(segments), len(window) // 2))
    spectral_time = numpy.empty(len(segments))

    for index, segment in enumerate(segments):
        istart = segment[0]
        iend = segment[1]
        spectral_time[index] = (epoch_time[istart] + epoch_time[iend]) / 2

        (
            e[index, :],
            a1[index, :],
            b1[index, :],
            a2[index, :],
            b2[index, :],
        ) = power_spectra(
            x[istart:iend],
            y[istart:iend],
            z[istart:iend],
            window,
            sampling_frequency,
            window_overlap_fraction,
        )

    return spectral_time, e, a1, b1, a2, b2


@njit(cache=True)
def window_power_correction_factor(window: numpy.ndarray) -> float:
    """
    Calculate the power correction factor that corrects for the loss in power
    due to the aplication of the windowing function.

    Basically for a DC signal we want the windowed mean power to be equal to the non-windowed power

    mean (   (1 * ScaledWindow) **2 )  = mean( 1 )

    whereas the raw window gives

    Power = mean (   (Window) **2 )

    Hence- ScaledWindow = Window * sqrt( 1 / mean(window**2) )

    :param window: Window as numpy ndarray
    :return:
    """
    return 1 / numpy.sqrt(numpy.mean(window**2))


def estimate_spectrum(
    epoch_time,
    x,
    y,
    z,
    window=None,
    segment_length_seconds=1800,
    window_overlap_fraction=0.5,
    sampling_frequency=2.5,
    depth=None,
    latitude=None,
    longitude=None,
) -> FrequencySpectrum:
    """

    :param epoch_time: epoch time in seconds
    :param x: East(positive)/west displacement
    :param y: North(positive)/south displacement
    :param z: vertical (up positive) displacement
    :param window: window function to apply. If None are given a 256 point hann window is applied (default on Spotter),
                   which corresponds to 102.4 second length window at a sampling frequency of 2.5 Hz
                   (default on Spotter)
    :param segment_length_seconds:  Segment length for the spectral analysis. If none is given a 1800
    :param window_overlap_fraction: Overlap between successive windows (default 0.5)
    :param sampling_frequency: sampling frequency in Hertz, default 2.5 Hz
    :param depth:
    :param latitude:
    :param longitude:
    :return:
    """

    if window is None:
        window = get_window("hann", 256)

    spectral_time, e, a1, b1, a2, b2 = welch(
        epoch_time,
        x,
        y,
        z,
        window,
        sampling_frequency,
        segment_length_seconds,
        window_overlap_fraction,
    )

    df = sampling_frequency / len(window)
    frequencies = (
        numpy.linspace(0, len(window) // 2, len(window) // 2, endpoint=True) * df
    )

    if depth is None:
        depth = numpy.full(e.shape[0], numpy.inf)

    if latitude is None:
        latitude = numpy.full(e.shape[0], numpy.nan)

    if longitude is None:
        longitude = numpy.full(e.shape[0], numpy.nan)

    dataset = Dataset(
        data_vars={
            "variance_density": (["time", "frequency"], e),
            "a1": (["time", "frequency"], a1),
            "b1": (["time", "frequency"], b1),
            "a2": (["time", "frequency"], a2),
            "b2": (["time", "frequency"], b2),
            "depth": (["time"], depth),
            "latitude": (["time"], latitude),
            "longitude": (["time"], longitude),
        },
        coords={"time": spectral_time.astype("<M8[s]"), "frequency": frequencies},
    )
    return FrequencySpectrum(dataset)
