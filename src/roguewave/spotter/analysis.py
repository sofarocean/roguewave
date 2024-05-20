import typing

import numpy as np
import pandas
import pandas as pd

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
from roguewave.wavespectra import concatenate_spectra
from roguewave.timeseries_analysis import estimate_frequency_spectrum
from roguewave.tools.time import datetime64_to_timestamp
from pandas import DataFrame, concat
from roguewave import FrequencySpectrum
from .read_csv_data import read_displacement, read_gps, read_rbr, read_baro
from .parser import apply_to_group
from numpy import real, conjugate, linspace
from roguewave.spotter._pressure_analysis import surface_elevation_from_pressure, sample_irregular_signal, frequency_scale
from roguewave.wavephysics.fluidproperties import WATER_DENSITY, GRAVITATIONAL_ACCELERATION
from scipy.signal import butter
import os
from roguewave.timeseries_analysis.filtering import sos_filter
from roguewave.spotter.parser import _netcdf_filename, save_as_netcdf
import xarray as xr

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

def surface_elevation_from_rbr(
        path,
        sensor_type,
        sensor_height=0,
        sampling_frequency=2,
        cache_as_netcdf=True,
        **kwargs):

    data = read_rbr(path, sensor_type=sensor_type, cache_as_netcdf=cache_as_netcdf, **kwargs)
    water_density = kwargs.get("density", WATER_DENSITY)
    gravitational_acceleration = kwargs.get("gravitational_acceleration", GRAVITATIONAL_ACCELERATION)

    dataframes = []
    for group in data['group id'].unique():
        data_group = data[ data['group id'] == group]
        pressure_head_meter = data_group['pressure (pascal)'].values / water_density / gravitational_acceleration
        irregular_time_in_seconds =data_group['time'].values.astype(float)/1e9

        if np.nanmean(data_group['pressure (pascal)'].values) < 0:
            continue

        time_in_seconds,p = sample_irregular_signal(
            irregular_time_in_seconds,
            pressure_head_meter,
            sampling_frequency
        )

        dataframe = surface_elevation_from_pressure(p,sampling_frequency,sensor_height=sensor_height,**kwargs)

        dataframe['time'] = pandas.to_datetime(time_in_seconds,unit='s')
        dataframe['group id'] = group
        dataframes.append(dataframe)

    return pd.concat(dataframes)


def surface_elevation_from_rbr_and_spotter(
        path,
        sensor_type,
        sensor_height=0,
        sampling_frequency_rbr=2,
        bandpass=True,
        cache_as_netcdf=True,
        **kwargs
):
    _str = f'_{sensor_type}_{sensor_height}_{sampling_frequency_rbr}_{bandpass}'
    for key in kwargs:
        _str += f'_{key}={kwargs[key]}'

    filename = f'{sensor_type}_and_spotter'+_str
    filepath = _netcdf_filename(path,filename  )
    if cache_as_netcdf and os.path.exists(filepath):
        return xr.open_dataset(filepath).to_dataframe()

    rbr = surface_elevation_from_rbr(path, sensor_type, sensor_height,sampling_frequency=sampling_frequency_rbr,
                                     cache_as_netcdf=cache_as_netcdf, **kwargs)

    spotter = read_displacement(path, cache_as_netcdf=cache_as_netcdf, postprocess=True)

    data = DataFrame()

    spotter_values = spotter['z'].values
    spotter_time = spotter['time'].values.astype(float) / 1e9

    rbr_time = rbr['time'].values.astype(float) / 1e9

    data['time'] = rbr['time']
    data['rbr surface elevation (meter)'] = rbr['surface elevation (meter)']
    data['spotter surface elevation (meter)'] = np.interp(rbr_time, spotter_time, spotter_values)
    data['depth (meter)'] = rbr['depth (meter)']
    data['group id'] = rbr['group id']

    if bandpass:
        data = _band_pass_filter(data, sampling_frequency_rbr, **kwargs)

    if cache_as_netcdf:
        save_as_netcdf(data, path, filename)
    return data


def _band_pass_filter(data, sampling_frequency, **kwargs):
    def func(data):
        df = pd.DataFrame()
        df['time'] = data['time']


        keys = ['rbr surface elevation (meter)', 'spotter surface elevation (meter)']
        df['depth (meter)'] = data['depth (meter)']
        depth = data['depth (meter)'].values[0]
        freq = 2.*frequency_scale(depth)

        if freq < sampling_frequency/2:
            sos = butter(4, [0.033,freq], btype='bandpass', fs=sampling_frequency, output='sos')
        else:
            sos = butter(4, 0.033, btype='highpass', fs=sampling_frequency, output='sos')
        for key in keys:
            if len(data[key].values) < 100:
                df[key] = data[key]
            else:
                df[key] = sos_filter(data[key].values, 'filtfilt', sos=sos)
        return df

    return apply_to_group(func, data)


def spectra_from_rbr_and_spotter(
        path,
        sensor_type,
        window_length,
        sensor_height=0,
        sampling_frequency_rbr=2,
        bandpass=False,
        cache_as_netcdf=True,
        **kwargs) -> typing.Tuple[FrequencySpectrum,FrequencySpectrum]:

    data = surface_elevation_from_rbr_and_spotter(
        path,
        sensor_type,
        sensor_height,
        sampling_frequency_rbr,
        bandpass=bandpass,
        cache_as_netcdf=cache_as_netcdf,
        **kwargs
    )

    time = data['time'].values.astype(float) / 1e9
    rbr = data['rbr surface elevation (meter)'].values
    spot = data['spotter surface elevation (meter)'].values

    x = np.zeros_like(rbr)
    y = np.zeros_like(rbr)

    spot_spectrum = estimate_frequency_spectrum(
        time,
        x,
        y,
        spot,
        segment_length_seconds=window_length,
        sampling_frequency=sampling_frequency_rbr,
        **kwargs
    )


    rbr_spectrum = estimate_frequency_spectrum(
        time,
        x,
        y,
        rbr,
        segment_length_seconds=window_length,
        sampling_frequency=sampling_frequency_rbr,
        **kwargs
    )


    return rbr_spectrum, spot_spectrum