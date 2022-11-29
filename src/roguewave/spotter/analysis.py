from roguewave.timeseries_analysis import pipeline, DEFAULT_SPOTTER_PIPELINE
from roguewave.timeseries_analysis.time_integration import (
    cumulative_distance,
    complex_response,
)
from roguewave.timeseries_analysis import estimate_frequency_spectrum
from roguewave.tools.time import datetime64_to_timestamp
from pandas import DataFrame
from roguewave import FrequencySpectrum
from .read_csv_data import read_displacement, apply_to_group, read_gps
from numpy import real, conjugate


def displacement_from_gps_doppler_velocities(
    path, pipeline_stages=None, **kwargs
) -> DataFrame:
    if pipeline_stages is None:
        pipeline_stages = DEFAULT_SPOTTER_PIPELINE

    doppler_velocities = read_gps(path)

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
    pipeline_stages = [
        "exponential_delta",
        "sos",
    ]

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


def spectra_from_raw_gps(path, **kwargs) -> FrequencySpectrum:
    displacement_doppler = displacement_from_gps_doppler_velocities(path, **kwargs)
    displacement_location = displacement_from_gps_positions(path)

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
