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
from numba.typed import Dict
from numba import types


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
    epoch_time: NDArray,
    signals: Tuple[NDArray],
    window: NDArray,
    options: Dict = None,
    spectral_window: NDArray = None,
) -> NDArray:

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    window_overlap_fraction = options.get("window_overlap_fraction", 0.5)
    sampling_frequency = options.get("sampling_frequency", 2.5)
    timebase_jitter_fraction = options.get("timebase_jitter_fraction", 2.5)

    overlap = int(len(window) * window_overlap_fraction)
    nyquist_index = len(window) // 2

    nsig = len(signals)
    nt = len(signals[0])
    output = numpy.zeros((nsig, nsig, nyquist_index), dtype="complex_")
    ffts = numpy.zeros((nsig, nyquist_index), dtype="complex_")

    zero_mean_signals = numpy.zeros((nsig, nt))
    for index, signal in enumerate(signals):
        zero_mean_signals[index, :] = signal - numpy.mean(signal)

    time_base = (epoch_time - epoch_time[0]) * sampling_frequency
    time_delta = numpy.diff(time_base)

    ii = 0
    istart = 0
    while True:
        # Advance the counter
        iend = istart + len(window)
        if iend > nt:
            break

        if numpy.any(time_delta[istart:iend] > 1 + timebase_jitter_fraction):
            istart = istart + len(window) - overlap
            continue

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

                if mm != nn:
                    output[nn, mm, :] = numpy.conjugate(output[mm, nn, :])

        istart = istart + len(window) - overlap

    if ii == 0:
        # No viable windows
        output[:, :, :] = numpy.nan
        return output

    frequency_step = sampling_frequency / len(window)
    output *= (
        window_power_correction_factor(window) ** 2
        / ii
        / frequency_step
        / len(window) ** 2
    )

    # Apply spectral smoothing if requested.
    if spectral_window is not None:
        pad_len = len(spectral_window) // 2
        for mm in range(0, nsig):
            for nn in range(mm, nsig):
                # Convolutional padding length
                total_energy_pre_smoothing = numpy.sum(output[mm, nn, :])
                if total_energy_pre_smoothing == 0.0:
                    continue

                # Window avereaged energy through convolution.
                smoothed = numpy.convolve(output[mm, nn, :], spectral_window)
                total_energy_post_smoothing = numpy.sum(smoothed[pad_len:-pad_len])

                # Renormalization to ensure we have the same pre/post total complex (co)variance
                renormalization_factor = (
                    total_energy_pre_smoothing / total_energy_post_smoothing
                )
                output[mm, nn, :] = renormalization_factor * smoothed[pad_len:-pad_len]

                # Update the opposite diagonal
                if mm != nn:
                    output[nn, mm] = numpy.conjugate(output[mm, nn])

    return output


@njit(cache=True)
def welch(
    epoch_time: numpy.ndarray,
    signals,
    window: numpy.ndarray,
    options: Dict = None,
    spectral_window=None,
) -> Tuple[NDArray, NDArray, NDArray]:

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    segment_length_seconds = options.get("segment_length_seconds", 1800.0)
    sampling_frequency = options.get("sampling_frequency", 2.5)
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
            epoch_time[istart:iend],
            segment_signal,
            window,
            options,
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


def estimate_frequency_spectrum(
    epoch_time: NDArray,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    window: NDArray = None,
    options=None,
    spectral_window=None,
    **kwargs
) -> FrequencySpectrum:
    """

    :param epoch_time:
    :param x:
    :param y:
    :param z:
    :param window:
    :param segment_length_seconds:
    :param window_overlap_fraction:
    :param sampling_frequency:
    :param depth:
    :param latitude:
    :param longitude:
    :param spectral_window:
    :param kwargs:
    :return:
    """
    if window is None:
        window = get_window("hann", 256)

    options = _to_numba_dict(options)

    spectral_time, frequencies, co_spectra = welch(
        epoch_time,
        (x, y, z),
        window,
        options,
        spectral_window,
    )

    szz, a1, b1, a2, b2 = extract_moments(co_spectra, index_x=0, index_y=1, index_z=2)

    dataset = Dataset(
        data_vars={
            "variance_density": (["time", "frequency"], szz),
            "a1": (["time", "frequency"], a1),
            "b1": (["time", "frequency"], b1),
            "a2": (["time", "frequency"], a2),
            "b2": (["time", "frequency"], b2),
            "depth": (["time"], numpy.full(szz.shape[0], numpy.inf)),
            "latitude": (["time"], numpy.full(szz.shape[0], numpy.nan)),
            "longitude": (["time"], numpy.full(szz.shape[0], numpy.nan)),
        },
        coords={"time": spectral_time.astype("<M8[s]"), "frequency": frequencies},
    )
    return FrequencySpectrum(dataset)


def _to_numba_dict(mapping) -> Dict:
    numba_mapping = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    if mapping is not None:
        for key, item in mapping.items():
            numba_mapping[key] = item

    return numba_mapping
