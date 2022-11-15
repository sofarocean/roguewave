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
import scipy.signal
import scipy.fft
from typing import List
from datetime import datetime, tzinfo, timezone
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput


class SpectralAnalysisConfig:
    def __init__(
        self,
        window_length=256,
        window_type="hann",
        window_overlap_percent=50,
        sampling_frequency_hertz=2.5,
    ):
        self.window_length = window_length
        self.window_type = window_type
        self.window_overlap_percent = window_overlap_percent
        self.sampling_frequency_hertz = sampling_frequency_hertz
        self.threshold = 10

    @property
    def window(self):
        return scipy.signal.windows.get_window(self.window_type, self.window_length)

    def scaled_window(self):
        # Return a window that corrects for power losses due to windowing and
        # incorporates the Fourier series constant
        return (
            window_power_correction_factor(self.window)
            / self.window_length
            / numpy.sqrt(self.bin_width)
        )

    @property
    def nyquist_index(self):
        return int(self.window_length / 2)

    @property
    def bin_width(self):
        return self.sampling_frequency_hertz / self.window_length

    @property
    def frequencies(self):
        return (
            numpy.linspace(0, 1, self.nyquist_index, endpoint=False)
            * self.sampling_frequency_hertz
            / 2
        )

    @property
    def overlapping_points(self):
        return int(self.window_length * self.window_overlap_percent / 100)

    def number_of_overlapping_windows(self, nfft):
        delta = self.window_length - self.overlapping_points
        return int(nfft / delta) - int(self.window_length / delta)


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


def calculate_moments(spectra):
    _sum = spectra("xx") + spectra["yy"]
    _diff = spectra("xx") - spectra["yy"]
    _root = numpy.sqrt(spectra("zz") * _sum)

    a1 = numpy.imag(spectra["xz"]) / _root
    b1 = numpy.imag(spectra["yz"]) / _root
    a2 = _diff / _sum
    b2 = 2 * numpy.real(spectra["xy"]) / _sum

    return a1, b1, a2, b2


def spectral_analysis(
    epoch_time: numpy.ndarray,
    x: numpy.ndarray,
    y: numpy.ndarray,
    z: numpy.ndarray,
    config: SpectralAnalysisConfig = None,
) -> WaveSpectrum1D:
    # ** Calculate constants related to the spectral analysis
    #
    def nans():
        return (
            numpy.zeros(
                (config.number_of_overlapping_windows, config.nyquist_index),
                dtype="complex_",
            )
            + numpy.nan
        )

    data = {
        "xx": nans(),
        "yy": nans(),
        "zz": nans(),
        "xz": nans(),
        "yz": nans(),
        "xy": nans(),
    }

    mapping = {
        "x": scipy.signal.detrend(x),
        "y": scipy.signal.detrend(y),
        "z": scipy.signal.detrend(z),
    }
    ii = -1
    istart = 0

    fft = {
        "x": numpy.zeros((config.nyquist_index,), dtype="complex_"),
        "y": numpy.zeros((config.nyquist_index,), dtype="complex_"),
        "z": numpy.zeros((config.nyquist_index,), dtype="complex_"),
    }

    has_valid_spectral_realization = False
    window = config.scaled_window()
    while True:
        # Advanve the counter
        ii += 1
        iend = istart + config.window_length
        if iend > len(epoch_time):
            break

        # Ensure contiguous timeseries in the window by checking if the maximum
        # difference in the time delta is consistent with the sampling frequency
        maximum_time_delta = numpy.max(numpy.diff(epoch_time[istart:iend]))
        if not 0.99 <= maximum_time_delta * config.sampling_frequency_hertz <= 1.01:
            # if not contiguous skip to the next window
            istart = istart + config.window_length - config.overlapping_points
            continue

        has_valid_spectral_realization = True

        # Calculate the fft for the three signals
        for key in mapping:
            fft[key] = scipy.fft.fft(mapping[key][istart:iend] * window)[
                0 : config.nyquist_index
            ]

        # Calculate the (co)spectral densities
        for keys in ("xx", "yy", "zz", "xy", "xz", "yz"):
            data[keys][ii, :] = fft[keys[0]] * numpy.conjugate(fft[keys[1]])

        istart = istart + config.window_length - config.overlapping_points

    if not has_valid_spectral_realization:
        return None

    median = numpy.nanmedian(data["zz"], axis=0)
    for ifreq in range(0, config.nyquist_index):
        mask = data["zz"][:, ifreq] > config.threshold * median[ifreq]
        for key in data:
            data[key][mask, ifreq] = numpy.nan

    spectra = {}
    for key in data:
        spectra[key] = numpy.nanmean(data[key], axis=0)

    a1, b1, a2, b2 = calculate_moments(spectra)

    time = datetime.fromtimestamp((epoch_time[0] + epoch_time[-1]) / 2, tz=timezone.utc)
    input = WaveSpectrum1DInput(
        frequency=config.frequencies,
        varianceDensity=spectra["zz"],
        timestamp=time,
        latitude=numpy.nan,
        longitude=numpy.nan,
        a1=a1,
        b1=b1,
        b2=b2,
        a2=a2,
    )
    return WaveSpectrum1D(input)


def generate_spectrum(
    epoch_time: numpy.ndarray,
    x: numpy.ndarray,
    y: numpy.ndarray,
    z: numpy.ndarray,
    segment_length_seconds=1800,
    config: SpectralAnalysisConfig = None,
) -> List[WaveSpectrum1D]:
    if not config:
        config = SpectralAnalysisConfig()

    assert len(x) == len(y) == len(z) == len(epoch_time)

    segments = segment_timeseries(epoch_time, segment_length_seconds)

    spectra = []

    for segment in segments:
        istart = segment[0]
        iend = segment[1]

        spectrum = spectral_analysis(
            epoch_time[istart:iend],
            x[istart:iend],
            y[istart:iend],
            z[istart:iend],
            config,
        )
        if spectrum is not None:
            spectra.append(spectrum)
    return spectra


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
