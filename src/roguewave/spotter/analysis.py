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
)
from roguewave.timeseries_analysis import estimate_frequency_spectrum
from roguewave.tools.time import datetime64_to_timestamp
from pandas import DataFrame, concat
from roguewavespectrum import Spectrum
from .read_csv_data import read_displacement, read_gps, read_rbr, read_baro
from .parser import apply_to_group
from roguewavespectrum.spotter._spotter_post_processing import spotter_frequency_response_correction
from roguewave.spotter._pressure_analysis import surface_elevation_from_pressure, sample_irregular_signal, frequency_scale
from roguewave.wavephysics.fluidproperties import WATER_DENSITY, GRAVITATIONAL_ACCELERATION
from scipy.signal import butter
import os
from roguewave.timeseries_analysis.filtering import sos_filter
from roguewave.timeseries_analysis.sampling import resample
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
) -> Spectrum:
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


def spectra_from_displacement(path, **kwargs) -> Spectrum:
    displacement = read_displacement(path,cache_as_netcdf=True)
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

    MINIMUM_SAMPLE_LENGTH = 900


    dataframes = []
    for group in data['group id'].unique():
        data_group = data[ data['group id'] == group]
        pressure_head_meter = data_group['pressure (pascal)'].values / water_density / gravitational_acceleration
        irregular_time_in_seconds =data_group['time'].values.astype(float)/1e9

        # If the pressure head is not a finite number/nan - exclude
        if not np.all(np.isfinite(pressure_head_meter)):
            continue

        # If depth is negative - exclude
        if np.nanmean(data_group['pressure (pascal)'].values) < 0:
            continue

        # If the sample is too short - exclude
        if len(irregular_time_in_seconds) < MINIMUM_SAMPLE_LENGTH:
            continue

        time_in_seconds,p = sample_irregular_signal(
            irregular_time_in_seconds,
            pressure_head_meter,
            sampling_frequency
        )

        dataframe = surface_elevation_from_pressure(p,sampling_frequency,sensor_height=sensor_height,**kwargs)

        # If the surface elevation is ever higher than the depth - exclude
        if dataframe['surface elevation (meter)'].max() > dataframe['depth (meter)'].max():
            continue

        # If the surface elevation is ever higher lower than the depth - exclude
        if dataframe['surface elevation (meter)'].min() < -dataframe['depth (meter)'].max():
            continue

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


    spotter_values_z = spotter['z'].values
    spotter_values_x = spotter['x'].values
    spotter_values_y = spotter['y'].values
    spotter_time = spotter['time'].values.astype(float) / 1e9

    rbr_time = rbr['time'].values.astype(float) / 1e9

    data['time'] = rbr['time']
    data['rbr surface elevation (meter)'] = rbr['surface elevation (meter)']
    data['spotter surface elevation (meter)'] = np.interp(rbr_time, spotter_time, spotter_values_z)
    data['spotter eastward displacement (meter)'] = np.interp(rbr_time, spotter_time, spotter_values_x)
    data['spotter northward displacement (meter)'] = np.interp(rbr_time, spotter_time, spotter_values_y)
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


        keys = ['rbr surface elevation (meter)', 'spotter surface elevation (meter)',
                'spotter eastward displacement (meter)', 'spotter northward displacement (meter)']
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
        **kwargs) -> typing.Tuple[Spectrum,Spectrum]:

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
        data['spotter eastward displacement (meter)'].values,
        data['spotter northward displacement (meter)'].values,
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