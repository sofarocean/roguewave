# Author: Pieter Bart Smit

"""
A set of spectral analysis routines specifically designed to work well with data
from Sofar Spotter ocean wave buoys.

Classes:

- `SpectralAnalysisConfig`, configuration object

Functions:

- `segment_timeseries`, split timeseries into non-overlapping segments
- `calculate_moments`, calculate directional moments from (co)variances
- `spectral_analysis`, calculate wave spectrum for specific segment
- `generate_spectra`, calculate wave spectra for all segments
- `window_power_correction_factor`, correct for power loss due to windowing

"""

from numba import njit, objmode, types
from numba.typed import Dict
import numpy
from numpy.fft import fft
from numpy.typing import NDArray
from roguewave import FrequencySpectrum
from scipy.signal.windows import get_window
from typing import List, Tuple, Mapping
from xarray import Dataset


def estimate_frequency_spectrum(
    epoch_time: NDArray,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    window: NDArray = None,
    segment_length_seconds=1800,
    sampling_frequency=2.5,
    options=None,
    spectral_window=None,
    response_functions=None,
    **kwargs
) -> FrequencySpectrum:
    """

    :param epoch_time:
    :param x:
    :param y:
    :param z:
    :param window:
    :param segment_length_seconds:
    :param sampling_frequency:
    :param options:
    :param spectral_window:
    :param kwargs:
    :return:
    """

    if window is None:
        window = get_window("hann", 256)

    options = _to_numba_dict(options)

    spectral_time, frequencies, co_spectra = estimate_co_spectra(
        epoch_time,
        (x, y, z),
        window,
        segment_length_seconds,
        sampling_frequency,
        options,
        spectral_window,
    )

    if response_functions is not None:
        for m in range(0, 3):
            for n in range(m, 3):
                response = response_functions[m](frequencies) * numpy.conjugate(
                    response_functions[n](frequencies)
                )
                co_spectra[:, m, n, :] = co_spectra[:, m, n, :] * response

                if m != n:
                    co_spectra[:, n, m, :] = co_spectra[:, m, n, :]

    szz, a1, b1, a2, b2 = _extract_moments(co_spectra, index_x=0, index_y=1, index_z=2)

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


@njit(cache=True)
def estimate_co_spectra(
    epoch_time: numpy.ndarray,
    signals,
    window: numpy.ndarray,
    segment_length_seconds,
    sampling_frequency,
    options: Dict = None,
    spectral_window=None,
) -> Tuple[NDArray, NDArray, NDArray]:
    segments = _segment_timeseries(epoch_time, segment_length_seconds)

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
            sampling_frequency,
            options,
            spectral_window,
        )

    df = sampling_frequency / len(window)
    frequencies = (
        numpy.linspace(numpy.int64(0), len(window) // 2 - 1, len(window) // 2) * df
    )

    return spectral_time, frequencies, co_spectra


# --------------------------
# Main spectral estimator
# --------------------------


@njit(cache=True)
def calculate_co_spectra(
    time_seconds: NDArray,
    signals: Tuple[NDArray],
    window: NDArray,
    sampling_frequency: float,
    options: Dict = None,
    spectral_window: NDArray = None,
) -> NDArray:
    """
    Calculate co-spectral density matrix using Welch's method.

    :param time_seconds:
    :param signals:
    :param window:
    :param sampling_frequency:
    :param options:
    :param spectral_window:
    :return:
    """

    if options is None:
        options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    window_overlap_fraction = options.get("window_overlap_fraction", 0.5)
    timebase_jitter_fraction = options.get("timebase_jitter_fraction", 2.5)

    # STEP 0: Preliminaries
    # ---------------------------------------------------------------------------

    # Signal and window lengths
    overlap = int(len(window) * window_overlap_fraction)
    nyquist_index = len(window) // 2
    nsig = len(signals)
    nt = len(signals[0])

    # Initialize output and work arrays
    output = numpy.zeros((nsig, nsig, nyquist_index), dtype="complex_")
    ffts = numpy.zeros((nsig, nyquist_index), dtype="complex_")
    zero_mean_signals = numpy.zeros((nsig, nt))

    # Correct the signals for mean contribution (detrend)
    for index, signal in enumerate(signals):
        zero_mean_signals[index, :] = signal - numpy.mean(signal)

    # Scale the time so that it runs as [0,1,...]. Note, while we may have some jitter (not exactly integeres) here
    # it is assumed there are no gaps in the signal.
    time_base = (time_seconds - time_seconds[0]) * sampling_frequency
    time_delta = numpy.diff(time_base)

    # initialize counters
    number_of_realizations = 0  # Number of valid realizations that fit in the signal
    istart = 0  # Start index of the window

    # STEP 1: Calculate windowed raw (unscaled) co-spectra
    # ---------------------------------------------------------------------------

    while True:

        # Find the end of the window
        iend = istart + len(window)

        # If the end index is larger than the length of the signal we are done
        if iend > nt:
            break

        # If there are any time delta's larger than the sample time step we reject the realization
        if numpy.any(time_delta[istart:iend] > 1 + timebase_jitter_fraction):
            istart = istart + len(window) - overlap
            continue

        # We have a "valid" realization
        number_of_realizations += 1

        # Calculate the fft of each of the signals
        for index in range(nsig):
            with objmode():
                # FFT not supported in Numba- yet
                ffts[index, :] = fft(zero_mean_signals[index, istart:iend] * window)[
                    0:nyquist_index
                ]

        # Calculate the co(spectral) contributions, making use of the Hermetian property of the co-spectral matrix,
        # i.e.   cospectra[mm,nn] == conjugate( cospectra[nn,mm] )
        for mm in range(0, nsig):
            for nn in range(mm, nsig):
                output[mm, nn, :] += 2 * ffts[mm, :] * numpy.conjugate(ffts[nn, :])

                if mm != nn:
                    output[nn, mm, :] = numpy.conjugate(output[mm, nn, :])

        # Advance the window to the next position
        istart = istart + len(window) - overlap

    # If no viable windows were found - return nan's
    if number_of_realizations == 0:
        output[:, :, :] = numpy.nan
        return output

    # STEP 2: Scale the co-spectra to an energy density output
    # ---------------------------------------------------------------------------

    # Multiply with scaling coeficient to account for:
    #  - power loss due to windowing -> * _window_power_correction_factor(window) ** 2
    #  - taking the average over all realizations -> / number_of_realizations
    #  - conversion to power density ->  / frequency_step
    #  - Fourier scaling constant of the FFT ->  / len(window) ** 2
    frequency_step = sampling_frequency / len(window)
    output *= (
        _window_power_correction_factor(window) ** 2
        / number_of_realizations
        / frequency_step
        / len(window) ** 2
    )

    # STEP 3: Convolutional smoothing in the frequency domain
    # ---------------------------------------------------------------------------

    # Apply spectral smoothing if requested.
    if spectral_window is not None:
        # Convolutional padding length
        pad_len = len(spectral_window) // 2

        # Do the smoothing for each of the signals, making use of the Hermetian property for efficiency
        for mm in range(0, nsig):
            for nn in range(mm, nsig):

                total_energy_pre_smoothing = numpy.sum(output[mm, nn, :])
                if total_energy_pre_smoothing == 0.0:
                    # In case of no energy (mostly for artificial or filtered input signals) we leave the signl
                    # unaltered (our renormalization would otherwise introduce a 0/0 -> nan
                    continue

                # Window avereaged energy through convolution.
                smoothed = numpy.convolve(output[mm, nn, :], spectral_window)

                # Renormalization to ensure we have the same pre/post total complex (co)variance
                total_energy_post_smoothing = numpy.sum(smoothed[pad_len:-pad_len])
                renormalization_factor = (
                    total_energy_pre_smoothing / total_energy_post_smoothing
                )
                output[mm, nn, :] = renormalization_factor * smoothed[pad_len:-pad_len]

                # Update the opposite diagonal with the conjugate of the signal
                if mm != nn:
                    output[nn, mm] = numpy.conjugate(output[mm, nn])

    return output


# ------------------
# Helper Functions
# ------------------


def _to_numba_dict(mapping: Mapping) -> Dict:
    """
    Convert a regular dictionary to a numba dictionary we can pass to jitted functions. Note that numba only supports
    Dicts that have the same type for the keys and the values for all entries.

    Purpose: mostly to make it easy to pass a normal python dict and convert it so something we can pass to
             numba.

    :param mapping: mapping to turn into a dict.
    :return:
    """

    numba_mapping = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    if mapping is not None:
        for key, item in mapping.items():
            numba_mapping[key] = item

    return numba_mapping


def _extract_moments(
    co_spectra: NDArray, index_x: int, index_y: int, index_z: int
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Extract the wave directional moments from a given co-spectral Matrix. Returns a tuple of numpy arrays as
    (szz,a1,b1,a2,b2)

    :param co_spectra: co-spectral matrix that contains (co)spectra of displacements.
    :param index_x: index of x in co-spectral array
    :param index_y: index of y in co-spectral array
    :param index_z: index of z in co-spectral array
    :return: Directional moments (szz,a1,b1,a2,b2)
    """

    # Vertical, and horizontal power spectra.
    szz = numpy.real(co_spectra[:, index_z, index_z, :])
    syy = numpy.real(co_spectra[:, index_y, index_y, :])
    sxx = numpy.real(co_spectra[:, index_x, index_x, :])

    # Co and quad spectra.
    cxy = numpy.real(co_spectra[:, index_x, index_y, :])
    qzx = numpy.imag(co_spectra[:, index_z, index_x, :])
    qzy = numpy.imag(co_spectra[:, index_z, index_y, :])

    # Calculat the directional moments from displacement (co) spectra
    a1 = qzx / numpy.sqrt(szz * (sxx + syy))
    b1 = qzy / numpy.sqrt(szz * (sxx + syy))
    a2 = (sxx - syy) / (sxx + syy)
    b2 = 2 * cxy / (sxx + syy)

    return szz, a1, b1, a2, b2


@njit(cache=True)
def _window_power_correction_factor(window: numpy.ndarray) -> float:
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


@njit(cache=True)
def _segment_timeseries(epoch_time, segment_length_seconds) -> List[tuple]:
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
