from roguewave.timeseries_analysis import (
    pipeline,
    DEFAULT_SPOTTER_PIPELINE,
    DEFAULT_DISPLACEMENT_PIPELINE,
)
from roguewave.tools.time_integration import (
    cumulative_distance,
    complex_response,
)
from roguewave.wavespectra.spectrum import fill_zeros_or_nan_in_tail
from roguewave.timeseries_analysis import estimate_frequency_spectrum
from roguewave.tools.time import datetime64_to_timestamp
from pandas import DataFrame
from roguewave import FrequencySpectrum
from .read_csv_data import read_displacement, read_gps
from .parser import apply_to_group
from numpy import real, conjugate, linspace

LAST_BIN_WIDTH = 0.3
LAST_BIN_FREQUENCY_START = 0.5
LAST_BIN_FREQUENCY_END = 0.8
SPOTTER_FREQUENCY_RESOLUTION = 2.5 / 256


def displacement_from_gps_doppler_velocities(
    path, pipeline_stages=None, cache_as_netcdf=False, **kwargs
) -> DataFrame:
    if pipeline_stages is None:
        pipeline_stages = DEFAULT_SPOTTER_PIPELINE

    doppler_velocities = read_gps(path, cache_as_netcdf=cache_as_netcdf)

    def process(data: DataFrame):
        time = data["time"].values
        x = pipeline(time, data["u"].values, pipeline_stages)
        y = pipeline(time, data["v"].values, pipeline_stages)
        z = pipeline(time, data["w"].values, pipeline_stages)

        return DataFrame(
            data={"time": time, "x": x, "y": y, "z": z, "group id": data["group id"]}
        )

    return apply_to_group(process, doppler_velocities)


def displacement_from_gps_positions(path) -> DataFrame:
    pipeline_stages = DEFAULT_DISPLACEMENT_PIPELINE

    gps_location_data = read_gps(path, postprocess=True)

    def get_cumulative_distances(data: DataFrame):
        time = data["time"].values
        x, y = cumulative_distance(data["latitude"].values, data["longitude"].values)
        z = data["z"].values

        return DataFrame(
            data={"time": time, "x": x, "y": y, "z": z, "group id": data["group id"]}
        )

    cum_distance = apply_to_group(get_cumulative_distances, gps_location_data)

    def process(data: DataFrame):
        time = data["time"].values
        x = pipeline(time, data["x"].values, pipeline_stages)
        y = pipeline(time, data["y"].values, pipeline_stages)
        z = pipeline(time, data["z"].values, pipeline_stages)

        return DataFrame(
            data={"time": time, "x": x, "y": y, "z": z, "group id": data["group id"]}
        )

    return apply_to_group(process, cum_distance)


def spectra_from_raw_gps(
    path=None, displacement_doppler=None, displacement_location=None, **kwargs
) -> FrequencySpectrum:
    if displacement_doppler is None:
        displacement_doppler = displacement_from_gps_doppler_velocities(path, **kwargs)

    if displacement_location is None:
        try:
            displacement_location = displacement_from_gps_positions(path)
        except:
            displacement_location = None

    correct_for_numerical_integration = kwargs.get("response_correction", True)
    order = kwargs.get("order", 4)
    n = kwargs.get("n", 1)

    if kwargs.get("use_u", False):
        x = displacement_doppler["x"].values
    else:
        x = displacement_location["x"].values

    if kwargs.get("use_v", False):
        y = displacement_doppler["y"].values
    else:
        y = displacement_location["y"].values

    if kwargs.get("use_w", True):
        z = displacement_doppler["z"].values
    else:
        z = displacement_location["z"].values

    time = datetime64_to_timestamp(displacement_doppler["time"].values)
    spectrum = estimate_frequency_spectrum(time, x, y, z, **kwargs)

    if correct_for_numerical_integration:
        spectrum = spotter_frequency_response_correction(spectrum, order, n)
    return spectrum


def spectra_from_displacement(path, **kwargs) -> FrequencySpectrum:
    displacement = read_displacement(path)
    time = datetime64_to_timestamp(displacement["time"].values)

    kwargs = kwargs.copy()
    correct_for_numerical_integration = kwargs.pop("response_correction", True)
    order = kwargs.pop("order", 4)
    n = kwargs.pop("n", 1)

    spectrum = estimate_frequency_spectrum(
        time,
        displacement["x"].values,
        displacement["y"].values,
        displacement["z"].values,
        **kwargs
    )
    if correct_for_numerical_integration:
        spectrum = spotter_frequency_response_correction(spectrum, order, n)
    return spectrum


def spotter_frequency_response_correction(
    spectrum: FrequencySpectrum, order=4, n=1, sampling_frequency=2.5
) -> FrequencySpectrum:
    """
    Correct for the spectral dampening/amplification caused by numerical integration of velocities.
    :param spectrum:
    :param order:
    :param n:
    :return:
    """
    amplification_factor = complex_response(
        spectrum.frequency.values / sampling_frequency, order, n
    )
    R = real(amplification_factor * conjugate(amplification_factor))
    return spectrum.multiply(1 / R, ["frequency"])


def spotter_api_spectra_post_processing(
    spectrum: FrequencySpectrum, maximum_frequency=LAST_BIN_FREQUENCY_END
):
    """
    Post processing to spectra obtained from the API.

    :param spectrum: input spectra.
    :param maximum_frequency: maximum frequency to extrapolate to.
    :return:
    """
    maximum_index = int(maximum_frequency / SPOTTER_FREQUENCY_RESOLUTION)
    new_frequencies = (
        linspace(3, maximum_index, maximum_index - 2, endpoint=True)
        * SPOTTER_FREQUENCY_RESOLUTION
    )

    if len(spectrum.frequency) == 39:
        new_frequencies[0] = spectrum.frequency[0]

        last_bin_energy = spectrum.variance_density.values[..., -1] * LAST_BIN_WIDTH
        last_bin_moments = {
            "a1": spectrum.a1[..., -1],
            "b1": spectrum.b1[..., -1],
            "a2": spectrum.a2[..., -1],
            "b2": spectrum.b2[..., -1],
        }

        spectrum = spectrum.interpolate_frequency(new_frequencies)
        spectrum = spectrum.bandpass(fmax=LAST_BIN_FREQUENCY_START)

        # Correct for integration errors in the tail
        spectrum = spotter_frequency_response_correction(spectrum)

        # Extrapolate tail given the known energy in last bin. Also correct for potential underflow in the tail
        spectrum = spectrum.extrapolate_tail(
            maximum_frequency,
            tail_energy=last_bin_energy,
            tail_bounds=(LAST_BIN_FREQUENCY_START, LAST_BIN_FREQUENCY_END),
            tail_moments=last_bin_moments,
            tail_frequency=new_frequencies[new_frequencies > LAST_BIN_FREQUENCY_START],
        )
    else:
        # Chop to desired max freq
        spectrum = spectrum.bandpass(fmax=maximum_frequency)

        # Correct for integration errors in the tail
        spectrum = spotter_frequency_response_correction(spectrum)

        # Correct for potential underflow in the tail
        spectrum = fill_zeros_or_nan_in_tail(spectrum)

        spectrum = spectrum.interpolate_frequency(new_frequencies)

    return spectrum
