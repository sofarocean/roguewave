from roguewave.timeseries_analysis import pipeline
from roguewave.timeseries_analysis.time_integration import cumulative_distance
from roguewave.timeseries_analysis import estimate_frequency_spectrum
from roguewave.tools.time import datetime64_to_timestamp
from pandas import DataFrame
from roguewave import FrequencySpectrum
from .read_csv_data import read_displacement, apply_to_group, read_gps


def displacement_from_gps_doppler_velocities(
    path, pipeline_stages=None, **kwargs
) -> DataFrame:
    if pipeline_stages is None:
        pipeline_stages = [
            ("spike", None),
            ("integrate", None),
            ("exponential_delta", None),
            ("sos_forward", None),
        ]

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
        "sos_filtfilt",
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
    return estimate_frequency_spectrum(time, x, y, z, **kwargs)


def spectra_from_displacement(path) -> FrequencySpectrum:
    displacement = read_displacement(path)
    time = datetime64_to_timestamp(displacement["time"].values)
    return estimate_frequency_spectrum(
        time,
        displacement["x"].values,
        displacement["y"].values,
        displacement["z"].values,
    )
