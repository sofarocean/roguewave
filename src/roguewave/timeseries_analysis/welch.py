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
from xarray import Dataset, DataArray
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
def calculate_co_spectra(
    signals,
    window: numpy.ndarray,
    sampling_frequency,
    window_overlap_fraction=0.5,
    spectral_window=None,
) -> NDArray:
    overlap = int(len(window) * window_overlap_fraction)
    nyquist_index = len(window) // 2

    nsig = len(signals)
    nt = len(signals[0])
    output = numpy.zeros((nsig, nsig, nyquist_index), dtype="complex_")
    ffts = numpy.zeros((nsig, nyquist_index), dtype="complex_")

    zero_mean_signals = numpy.zeros((nsig, nt))
    for index, signal in enumerate(signals):
        zero_mean_signals[index, :] = signal - numpy.mean(signal)

    ii = 0
    istart = 0
    while True:
        # Advance the counter
        iend = istart + len(window)
        if iend > nt:
            break

        for index in range(nsig):
            with objmode():
                # FFT not supported in Numba- yet
                ffts[index, :] = fft(zero_mean_signals[index, istart:iend] * window)[
                    0:nyquist_index
                ]

        sigs = numpy.zeros((nsig, nyquist_index))
        for index in range(nsig):
            sigs[index, :] = numpy.real(
                ffts[index, :] * numpy.conjugate(ffts[index, :])
            )

        ii += 1
        for mm in range(0, nsig):
            for nn in range(mm, nsig):
                output[mm, nn, :] += 2 * ffts[mm, :] * numpy.conjugate(ffts[nn, :])

        istart = istart + len(window) - overlap

    frequency_step = sampling_frequency / len(window)
    factor = (
        window_power_correction_factor(window) ** 2
        / ii
        / frequency_step
        / len(window) ** 2
    )

    for mm in range(0, nsig):
        for nn in range(mm, nsig):

            if spectral_window is not None:
                scaling = numpy.sum(output[mm, nn, :])

                if not (numpy.abs(scaling) == 0.0):
                    n = len(spectral_window) // 2
                    smoothed = numpy.convolve(output[mm, nn, :], spectral_window)

                    output[mm, nn, :] = (
                        scaling * smoothed[n:-n] / numpy.sum(smoothed[n:-n])
                    )

            output[mm, nn, :] = output[mm, nn, :] * factor

            if mm != nn:
                output[nn, mm] = output[mm, nn]

    return output


@njit(cache=True)
def welch(
    epoch_time: numpy.ndarray,
    signals,
    window: numpy.ndarray,
    sampling_frequency: float,
    segment_length_seconds=1800,
    window_overlap_fraction=0.5,
    spectral_window=None,
) -> Tuple[NDArray, NDArray, NDArray]:
    segments = segment_timeseries(epoch_time, segment_length_seconds)

    number_of_spectra = len(segments)
    number_of_frequencies = len(window) // 2

    nsig = len(signals)
    co_spectra = numpy.empty(
        (number_of_spectra, nsig, nsig, number_of_frequencies), dtype="complex_"
    )
    spectral_time = numpy.empty(number_of_spectra)

    for index, segment in enumerate(segments):
        istart = segment[0]
        iend = segment[1]
        spectral_time[index] = (epoch_time[istart] + epoch_time[iend]) / 2

        segment_signal = [sig[istart:iend] for sig in signals]
        co_spectra[index, :, :, :] = calculate_co_spectra(
            segment_signal,
            window,
            sampling_frequency,
            window_overlap_fraction,
            spectral_window,
        )

    df = sampling_frequency / len(window)
    frequencies = (
        numpy.linspace(numpy.int64(0), len(window) // 2 - 1, len(window) // 2) * df
    )

    return spectral_time, frequencies, co_spectra


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


def extract_moments(co_spectra, index_x, index_y, index_z):
    szz = numpy.real(co_spectra[:, index_z, index_z, :])
    syy = numpy.real(co_spectra[:, index_y, index_y, :])
    sxx = numpy.real(co_spectra[:, index_x, index_x, :])
    cxy = numpy.real(co_spectra[:, index_x, index_y, :])
    qzx = numpy.imag(co_spectra[:, index_z, index_x, :])
    qzy = numpy.imag(co_spectra[:, index_z, index_y, :])

    a1 = qzx / numpy.sqrt(szz * (sxx + syy))
    b1 = qzy / numpy.sqrt(szz * (sxx + syy))
    a2 = (sxx - syy) / (sxx + syy)
    b2 = 2 * cxy / (sxx + syy)
    return szz, a1, b1, a2, b2


def estimate_co_spectra(
    epoch_time,
    signals,
    window=None,
    segment_length_seconds=1800,
    window_overlap_fraction=0.5,
    sampling_frequency=2.5,
    spectral_window=None,
) -> Tuple[NDArray, NDArray, NDArray]:
    if window is None:
        window = get_window("hann", 256)

    return welch(
        epoch_time,
        signals,
        window,
        sampling_frequency,
        segment_length_seconds,
        window_overlap_fraction,
        spectral_window,
    )


def estimate_frequency_spectrum(
    epoch_time,
    x,
    y,
    z,
    window=None,
    segment_length_seconds=1800,
    window_overlap_fraction=0.5,
    sampling_frequency=2.5,
    depth: DataArray = None,
    latitude: DataArray = None,
    longitude: DataArray = None,
    spectral_window=None,
    **kwargs
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
    spectral_time, frequencies, co_spectra = estimate_co_spectra(
        epoch_time,
        (x, y, z),
        window,
        segment_length_seconds,
        window_overlap_fraction,
        sampling_frequency,
        spectral_window,
    )

    szz, a1, b1, a2, b2 = extract_moments(co_spectra, index_x=0, index_y=1, index_z=2)

    if depth is None:
        depth = numpy.full(szz.shape[0], numpy.inf)
    else:
        depth = numpy.interp(spectral_time, depth["time"].values, depth.values)

    if latitude is None:
        latitude = numpy.full(szz.shape[0], numpy.nan)
    else:
        depth = numpy.interp(spectral_time, latitude["time"].values, latitude.values)

    if longitude is None:
        longitude = numpy.full(szz.shape[0], numpy.nan)
    else:
        depth = numpy.interp(spectral_time, longitude["time"].values, longitude.values)

    dataset = Dataset(
        data_vars={
            "variance_density": (["time", "frequency"], szz),
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
