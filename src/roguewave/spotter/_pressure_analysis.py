import numpy as np
from numpy.fft import rfft, irfft

from roguewave.timeseries_analysis.filtering import sos_filter
from scipy.signal import butter
from scipy.signal.windows import tukey
from roguewave.wavephysics.fluidproperties import GRAVITATIONAL_ACCELERATION
import pandas as pd


def frequency_scale(depth, **kwargs):
    """
    Calculate the frequency scale for the response function. This frequency scale is simply derived from the shallow
    water relation - noting that

    $$ kd=1  $$

    corresponds to

    $$ f = \sqrt{g/d} / 2\pi $$

    if we use the shallow water dispersion relation

    $$ \omega^2 = gd k^2 $$

    :param depth: depth in meters
    :return: frequency scale in Hz
    """
    g = kwargs.get("g", GRAVITATIONAL_ACCELERATION)
    return np.sqrt(depth / g) / (2 * np.pi)


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
    interpolated_signal = np.interp(interpolated_time_base, time_base, signal)
    return interpolated_time_base / sampling_frequency + time_seconds[0], interpolated_signal


def spectral_derivative(signal, sampling_frequency, sos=None):
    """
    Calculate the time derivative of a signal using the Fourier transform. The signal is assumed to be sampled at a
    constant rate given by the sampling frequency.

    There will be edge effects unless the signal has been windowed.

    :param signal: signal
    :param sampling_frequency: sampling frequency in Hz
    :return: time derivative of the signal
    """
    if sos is not None:
        signal = sos_filter(signal, 'filtfilt', sos=sos)

    n = len(signal)
    omega = 2 * np.pi * np.fft.rfftfreq(n, 1 / sampling_frequency)

    return irfft(1j * omega * rfft(signal, n), n)


def _non_linear_contribution(pressure_head_meter, sampling_frequency, sensor_height, **kwargs):
    """
    Calculate the non-linear contribution to the pressure signal. The non-linear contribution has the form

    $$ \\frac{1}{2g} \\frac{d}{dt} \\left( h \\frac{d h}{dt} \\right) $$

    where $h$ is the pressure head in meters. The signal is assumed to be sampled at a constant rate given by the
    sampling frequency.

    :param pressure_head_meter: Water pressure head in meter. The signal is assumed to be sampled at a constant rate
        given by the sampling frequency.
    :param sampling_frequency: Sampling frequency in Hz.
    :param sensor_height: Height (in meters) of the sensor above the bed level.

    :return: Non-linear contribution to the surface elevation in meters.
    """
    _filter = kwargs.get("filter", True)
    g = kwargs.get("gravitational_acceleration", GRAVITATIONAL_ACCELERATION)
    mean_depth = np.nanmean(pressure_head_meter) + sensor_height

    if _filter:
        relative_cutoff_frequency = kwargs.get("relative_cutoff_frequency", 2)
        cutoff_frequency = frequency_scale(mean_depth, **kwargs) * relative_cutoff_frequency

        if cutoff_frequency < sampling_frequency / 2:
            sos = kwargs.get(
                "sos",
                butter(4, cutoff_frequency, btype='lowpass', fs=sampling_frequency, output='sos')
            )
            pressure_head_meter = sos_filter(pressure_head_meter, 'filtfilt', sos=sos)

    hydrostatic_estimate = pressure_head_meter - np.nanmean(pressure_head_meter)
    dpdt = spectral_derivative(hydrostatic_estimate, sampling_frequency)
    return spectral_derivative(hydrostatic_estimate * dpdt, sampling_frequency) / 2 / g


def slope_term(pressure_in_meters, sampling_frequency, sensor_height=0,**kwargs):
    slope = kwargs.get("slope", 0)
    n = len(pressure_in_meters)

    mean_depth = np.nanmean(pressure_in_meters) + sensor_height
    omega = 2 * np.pi * np.fft.rfftfreq(n, 1 / sampling_frequency)
    g = kwargs.get("gravitational_acceleration", GRAVITATIONAL_ACCELERATION)
    amplification_factor = 1j * omega * slope * np.sqrt( g / mean_depth)/2

    return irfft(rfft(pressure_in_meters, n) * amplification_factor, n)


def non_hydrostatic_contribution(pressure_in_meters, sampling_frequency, sensor_height=0, **kwargs):
    """
    Calculate the non-hydrostatic contribution to the pressure signal. Here we use a truncated Taylor series expansion
    of the full expression.

    :param pressure_in_meters: Water pressure in meters. The signal is assumed to be sampled at a constant rate given by
        the sampling frequency.
    :param sampling_frequency: Sampling frequency in Hz.
    :param sensor_height: Height (in meters) of the sensor above the bed level.

    :return: Non-hydrostatic contribution to the surface elevation in meters.
    """
    _filter = kwargs.get("filter", True)
    g = kwargs.get("gravitational_acceleration", GRAVITATIONAL_ACCELERATION)
    current = kwargs.get('current_ms', 0)
    current_angle = kwargs.get('wave_current_mutual_angle_rad', 0)
    mean_depth = np.nanmean(pressure_in_meters) + sensor_height

    if _filter:
        relative_cutoff_frequency = kwargs.get("relative_cutoff_frequency", 2)
        cutoff_frequency = frequency_scale(mean_depth, **kwargs) * relative_cutoff_frequency

        if cutoff_frequency < sampling_frequency / 2:
            sos = kwargs.get(
                "sos",
                butter(4, cutoff_frequency, btype='lowpass', fs=sampling_frequency, output='sos')
            )
            pressure_in_meters = sos_filter(pressure_in_meters, 'filtfilt', sos=sos)

    order = kwargs.get("order", 4)
    taper = kwargs.get("taper", True)

    n = len(pressure_in_meters)
    omega = 2 * np.pi * np.fft.rfftfreq(n, 1 / sampling_frequency)
    depth = np.nanmean(pressure_in_meters) + sensor_height

    relative_omega = omega * np.sqrt(depth / g) * np.sqrt((depth ** 2 - sensor_height ** 2) / (depth ** 2))
    if taper:
        relative_omega = np.sqrt(5 * np.tanh(relative_omega ** 2 / 5))

    relative_current = np.cos(current_angle)*current / np.sqrt(g * depth)

    fac = np.ones(4)
    fac[order:] = 0
    amplification_factor = (
            fac[1] * relative_omega ** 2 / 2 +
            fac[2] * 5 * relative_omega ** 4 / 24 +
            fac[3] * 37 * relative_omega ** 6 / 720 -
            relative_current * relative_omega ** 2
    )

    return irfft(rfft(pressure_in_meters, n) * amplification_factor, n)


def surface_elevation_from_pressure(
        pressure_head_meter,
        sampling_frequency,
        sensor_height=0,
        **kwargs
) -> pd.DataFrame:
    """
    Calculate the surface elevation from a pressure head signal (pressure divided by water density and gravitational
    acceleration). The signal is assumed to be sampled at a constant rate given by the sampling frequency.

    :param pressure_head_meter: Water pressure head in meter. The signal is assumed to be sampled at a constant rate
        given by the sampling frequency.
    :param sampling_frequency: Sampling frequency in Hz.
    :param sensor_height: Height (in meters) of the sensor above the bed level.
    :return: Zero-mean Surface elevation in meters.
    """

    # Window
    # ------
    # Because we use Fourier transforms to calculate the non-hydrostatic/nonlinear contributions, we need to window the
    # signal to avoid edge effects. The default window is a Tukey window that tapers the nonhydrostatic/nonlinear signal
    # to zero at the edges.
    number_of_taper_points = kwargs.get("number_of_taper_points", 20)
    taper_fraction_in_default_window = number_of_taper_points / len(pressure_head_meter)
    window = kwargs.get("window", tukey(len(pressure_head_meter),taper_fraction_in_default_window))

    # Calculate mean water levels
    # ---------------------------
    mean_pressure_head = np.nanmean(pressure_head_meter)
    mean_depth = mean_pressure_head + sensor_height

    # Hydrostatic estimate
    # --------------------
    dataframe = pd.DataFrame()
    dataframe['surface elevation (meter)'] = pressure_head_meter - mean_pressure_head
    dataframe['depth (meter)'] = mean_depth

    # For very short signals we cannot do Fourier analysis- return the hydrostatic estimate
    if len(pressure_head_meter) < 100:
        return dataframe

    # Non-Hydrostatic estimate
    # ------------------------
    dataframe['surface elevation (meter)'] += window * non_hydrostatic_contribution(
        pressure_head_meter, sampling_frequency, sensor_height, **kwargs)

    # Slope estimate
    # ------------------------
    dataframe['surface elevation (meter)'] += window * slope_term(
        pressure_head_meter, sampling_frequency, sensor_height, **kwargs)

    # Nonlinear estimate
    # ------------------
    if kwargs.get("nonlinear_contribution", False):
        dataframe['surface elevation (meter)'] += (
                _non_linear_contribution(pressure_head_meter, sampling_frequency, sensor_height) * window)

    return dataframe
