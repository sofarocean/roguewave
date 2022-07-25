"""
Contents: Routines to get spectral data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to get spectral data from the spotter api

Functions:

- `get_spectrum`, function to call externally do download
                                         spectral data. Parallel if multiple Spotters
                                         are given.
- `_get_next_100_spectra`, Internal helper function. Grabs the next 100 spectra
                          from the given start data. Needed because the Spotter_API
                          will return a maximum of 100 spectra per call.
- '_download_spectra', function that abstracts away the limitation of 100 spectra
                       per call. Handles updating the start date etc.

How To Use This Module
======================
(See the individual functions for details.)

1. Import it: ``import roguewave.externaldata.spotterapi``
2. call ``get_spectrum`` to get the desired data.

TOC
======================

1) Imports: all external import data
2) Constants: all constants used in the module
3) Interfaces: overloaded interfaces for the main function (handles typing, no
               actual logic)
4) Main function: implementation of the main function
    - get_spectrum
5) Private Functions: module private functions that are used by the main function:
    - _get_next_100_spectra
    - _download_spectra
"""

# 1) Imports
# ======================
from datetime import datetime, timedelta, timezone
from .exceptions import \
    ExceptionNoDataForVariable
from multiprocessing.pool import ThreadPool
from pysofar.spotter import Spotter, SofarApi
from roguewave import logger
from roguewave.tools import datetime_to_iso_time_string, to_datetime
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, \
    WaveSpectrum1DInput
from roguewave.metoceandata import WaveBulkData, as_dataframe, WindData, \
    SSTData, MetoceanData, BarometricPressure
from typing import Dict, List, Union, overload, TypedDict, Tuple
from tqdm import tqdm
from pandas import DataFrame
import numpy

API = None

# 2) Constants
# ======================

# Maximum number of spectra to retrieve from the Spotter API per API call. Note
# that 2- os a hard limit of the API. If set higher than 100 it will just return
# 100 (and the implementation will fail)
MAX_LOCAL_LIMIT = 100
MAX_LOCAL_LIMIT_BULK = 500

# Maximum number of workers in the Threadpool. Should be set to something reasonable
# to not overload wavefleet
MAXIMUM_NUMBER_OF_WORKERS = 40
NUMBER_OF_RETRIES = 2


# 2.5 API Return JSON format:
class ApiWaveData(TypedDict):
    significantWaveHeight: float
    peakPeriod: float
    meanPeriod: float
    peakDirection: float
    peakDirectionalSpread: float
    meanDirection: float
    meanDirectionalSpread: float
    timestamp: str
    latitude: float
    longitude: float


class ApiWindData(TypedDict):
    speed: float
    direction: float
    timestamp: str
    latitude: float
    longitude: float


class ApiSSTData(TypedDict):
    degrees: float
    timestamp: str
    latitude: float
    longitude: float


class VariablesToInclude(TypedDict):
    frequencyData: bool
    waves: bool
    wind: bool
    surfaceTemp: bool
    barometerData: bool


# 3) Interfaces
# ======================
# these overloads only exist so we can make use of typing and autocomplete in
# e.g. Pycharm. Specifically, it handles the case that the return type depends
# on the input.
@overload
def get_spectrum(
        spotter_ids: List[str],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> Dict[str, List[WaveSpectrum1D]]: ...


@overload
def get_spectrum(
        spotter_ids: str,
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> List[WaveSpectrum1D]: ...


@overload
def get_bulk_wave_data(
        spotter_ids: List[str],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> Dict[str, List[WaveBulkData]]: ...


@overload
def get_bulk_wave_data(
        spotter_ids: str,
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> List[WaveBulkData]: ...


# 4) Main Function
# ======================
# Implements the interfaces above.
def get_sofar_api():
    if API is None:
        return SofarApi()
    else:
        return API


def get_spotter_ids(sofar_api: SofarApi = None) -> List[str]:
    if sofar_api is None:
        sofar_api = get_sofar_api()
    return sofar_api.device_ids


def get_spectrum(
        spotter_ids: Union[str, List[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> Union[Dict[str, List[WaveSpectrum1D]], List[WaveSpectrum1D]]:
    """
    Grabs the requested spectra for the spotter(s) in the given interval

    :param spotter_ids:
        Can be either 1) a List of spotter_ids or 2) a single Spotter_id.
            If a List of Spotters, i.e.: List[str]:
                the return type is a Dictionary, with the spotter_id as key and
                the dictionary entry a list of spectra for that Spotter in the
                requested time frame -> Dict[str,List[WaveSpectrum1D]]

            If a single spotter, i.e.: str:
                the return type is a Dictionary, with the spotter_id as key and
                the dictionary entry a list of spectra for that Spotter in the
                requested time frame. List[WaveSpectrum1D]

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :return: Data as a FrequencyDataList Object
    """
    data = get_data(
        spotter_ids,
        start_date,
        end_date,
        include_frequency_data=True,
        include_directional_moments=True,
        include_waves=False,
        include_wind=False,
        include_surface_temp_data=False,
        session=session,
        parallel_download=parallel_download,
    )
    out = {}
    for key in data:
        out[key] = data[key]['frequencyData']
    return out


def get_bulk_wave_data(
        spotter_ids: Union[str, List[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True
) -> Union[Dict[str, List[WaveBulkData]], List[WaveBulkData]]:
    """
    Grabs the requested spectra for the spotter(s) in the given interval

    :param spotter_ids:
        Can be either 1) a List of spotter_ids or 2) a single Spotter_id.
            If a List of Spotters, i.e.: List[str]:
                the return type is a Dictionary, with the spotter_id as key and
                the dictionary entry a list of spectra for that Spotter in the
                requested time frame -> Dict[str,List[WaveSpectrum1D]]

            If a single spotter, i.e.: str:
                the return type is a Dictionary, with the spotter_id as key and
                the dictionary entry a list of spectra for that Spotter in the
                requested time frame. List[WaveSpectrum1D]

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :return: Data as a FrequencyDataList Object
    """
    data = get_data(
        spotter_ids,
        start_date,
        end_date,
        include_frequency_data=False,
        include_directional_moments=False,
        include_waves=True,
        include_wind=False,
        include_surface_temp_data=False,
        session=session,
        parallel_download=parallel_download
    )
    out = {}
    for key in data:
        out[key] = data[key]['waves']
    return out


def get_data(
        spotter_ids: Union[str, List[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        include_frequency_data=False,
        include_directional_moments=False,
        include_waves=True,
        include_wind=False,
        include_barometer_data=False,
        include_surface_temp_data=False,
        session: SofarApi = None,
        parallel_download=True,
        bulk_data_as_dataframe=True
) -> Dict[str, Dict[
    str, Union[list[WaveSpectrum1D], list[WaveBulkData], DataFrame]]]:
    if spotter_ids is None:
        spotter_ids = get_spotter_ids()

    variables_to_include = VariablesToInclude(
        waves=include_waves,
        wind=include_wind,
        surfaceTemp=include_surface_temp_data,
        frequencyData=include_frequency_data,
        barometerData=include_barometer_data,
    )

    if not isinstance(spotter_ids, list):
        spotter_ids = [spotter_ids]
        return_list = False
    else:
        return_list = True

    if session is None:
        session = SofarApi()

    def worker(spotter_id):
        data = _download_data(spotter_id, session, variables_to_include,
                              start_date, end_date, bulk_data_as_dataframe)
        return data

    if parallel_download:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            out = list(
                tqdm(pool.imap(worker, spotter_ids),
                     total=len(spotter_ids)))
    else:
        out = list(
            tqdm(map(worker, spotter_ids), total=len(spotter_ids)))

    data = {}
    for spotter_id, spotter_data in zip(spotter_ids, out):
        #
        # Did we get any data for this spotter
        if spotter_data is not None:
            data[spotter_id] = spotter_data

    if not return_list:
        return data[spotter_ids[0]]
    else:
        return data


# 5) Private Functions
# ======================
def _download_data(
        spotter_id: str,
        session: SofarApi,
        variables_to_include: VariablesToInclude,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        bulk_data_as_dataframe: bool = True,
        limit: int = None) -> Dict[
    str, Union[List[WaveSpectrum1D], List[WaveBulkData]]]:
    """
    Function that downloads data from the API for the requested Spotter
    It abstracts away the limitation that the API can only return a maximum
    of 100 Spectra or 500 bulk data points for a single Spotter per call.

    :param spotter_id: ID for the Spotter we want to download
    :param session: Active session of the SofarAPI
    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history
    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to change
                  it. (calling function does not set the limit)

    :return: List of available data in the requested timeframe.
    """

    # Create a Spotter object to Query
    spotter = Spotter(spotter_id, spotter_id, session=session)

    # Set the initial start date. This will get advanced for every 100 spectra
    # we download
    _start_date = start_date

    if variables_to_include['frequencyData']:
        max_local_limit = MAX_LOCAL_LIMIT
    else:
        max_local_limit = MAX_LOCAL_LIMIT_BULK

    data = {}
    while True:
        # We can only download a maximum of 100 spectra at a time; so we need
        # to loop our request. We do not know how many spotters there are
        # in the given timeframe.
        #
        # Assumptions:
        #   - spotter api returns a maximum of N items per requests (N=500 if
        #     no spectral data, N=100 if so)
        #   - requests returned start from the requested start data and
        #     with the last entry either being the last entry that fits
        #     in the requested window, or merely the last sample that fits
        #     in the 100 items.
        #   - requests returned are in order

        number_of_items_returned = 0
        if limit is not None:
            # If we hit the limit (if given) of spotters requested, break
            if len(data) >= limit:
                break

            # otherwise, update our next request so we don't overrun
            # the limit
            local_limit = min(limit - len(data), max_local_limit)
        else:
            # if no limit is given, just ask for the maximum allowed.
            local_limit = max_local_limit

        try:
            # Try to get the next batch of data
            next, number_of_items_returned, max_timestamp = \
                _get_next_page(spotter, variables_to_include,
                               _start_date, end_date, local_limit)

        except ExceptionNoDataForVariable as e:
            # Could not download data, add nothing, raise warning
            raise e

        # Add the data to the list

        for key in next:
            if key in data:
                data[key] += next[key]
            else:
                data[key] = next[key]

        # If we did not receive all data we requested...
        if number_of_items_returned < local_limit:
            # , we are done...
            break
        else:
            # ... else we update the startdate to be the timestamp of the last
            # known entry we received plus a second, and use this as the new
            # start.
            _start_date = max_timestamp + timedelta(seconds=1)

    # Postprocessing
    if len(data) < 1:
        data = None
    else:
        # Convert bulk data to a dataframe if desired
        for key in data:
            if key == 'frequencyData':
                # We cannot convert the list of wavespectra to a dataframe.
                continue
            data[key] = as_dataframe(data[key])

    return data


def _get_next_page(
        spotter: Spotter,
        variables_to_include: VariablesToInclude,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        limit: int = MAX_LOCAL_LIMIT,
) -> Tuple[
    Dict[str, Union[List[WaveSpectrum1D], List[WaveBulkData]]], int, datetime]:
    """
    Function that downloads the page of Data from the Spotter API that lie
    within the given interval, starting from the record closest to the startdate.

    idiosyncrasies to handle:
    - Wavefleet sometimes returns multiple instances of the same record (same
      timestamp). These are filtered through a all to _unique. This should
      be fixed in the future.
    - Some entries are broken (None for entries).
    - Not all Spotters will have (all) data for the given timerange.
    - Wavefleet sometimes times out on requests.

    :param spotter: Spotter object from pysofar

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history

    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to change
                  it. (calling function does not set the limit)

    :return: Data
    """

    # Retry mechanism
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)
    json_data = None
    for retry in range(0, NUMBER_OF_RETRIES + 1):
        try:
            json_data = spotter.grab_data(
                limit=limit,
                start_date=datetime_to_iso_time_string(start_date),
                end_date=datetime_to_iso_time_string(end_date),
                include_frequency_data=variables_to_include[
                    'frequencyData'],
                include_directional_moments=variables_to_include[
                    'frequencyData'],
                include_barometer_data=variables_to_include['barometerData'],
                include_waves=variables_to_include['waves'],
                include_wind=variables_to_include['wind'],
                include_surface_temp_data=variables_to_include[
                    'surfaceTemp'],
            )
            break
        except Exception as e:
            if retry < NUMBER_OF_RETRIES:
                warning = f'Error downloading data for {spotter.id}, attempting retry {retry + 1}'
                logger.warning(warning)
            else:
                raise Exception

    out = {}
    max_num_items = 0
    max_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
    #
    # Loop over all possible variables
    for var_name, include_variable in variables_to_include.items():
        #
        # Did we want to include this variable?
        if include_variable:
            #
            # If so- was it returned? If not- raise error
            if var_name not in json_data:
                raise ExceptionNoDataForVariable(var_name)

            # If so- were any elements returned for this period? If not continue
            # We could error here, but there may be gaps in data for certain
            # sensors, so we try to continue
            if not json_data[var_name]:
                continue

            elements_returned = len(json_data[var_name])

            # Filter for Nones.
            json_data[var_name] = _none_filter(json_data[var_name])
            if not json_data[var_name]:
                # If after filtering for bad data nothing remains, continue.
                continue

            # How many items were returned (counting doubles and none values,
            # this is used for the pagination logic which only knows if it is
            # done if less then the requested data was returned).
            max_num_items = max(max_num_items, elements_returned)

            # Filter for doubles.
            json_data[var_name] = _unique_filter(json_data[var_name])

            # Add to output
            out[var_name] = \
                [_get_class(var_name, data) for data in json_data[var_name]]

            if out[var_name][-1].timestamp > max_timestamp:
                max_timestamp = out[var_name][-1].timestamp

    return out, max_num_items, max_timestamp


def search_circle(
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        center_lat_lon: Tuple,
        radius: float,
        session: SofarApi=None,
        variables_to_include: VariablesToInclude = None,
        page_size=500,
        bulk_data_as_dataframe: bool = True
):
    geometry = {'type': 'circle', 'points': center_lat_lon, 'radius': radius}

    return search(start_date, end_date, geometry, session,
                  variables_to_include, page_size, bulk_data_as_dataframe)


def search_rectangle(
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        north_west_lat_lon,
        south_east_lat_lon,
        session: SofarApi=None,
        variables_to_include: VariablesToInclude = None,
        page_size=500,
        bulk_data_as_dataframe: bool = True
):
    geometry = {'type': 'envelope',
                'points': [north_west_lat_lon, south_east_lat_lon],
                'radius': None}

    return search(start_date, end_date, geometry, session,
                  variables_to_include, page_size, bulk_data_as_dataframe)


def search(start_date: Union[datetime, str],
           end_date: Union[datetime, str],
           geometry: dict,
           session: SofarApi=None,
           variables_to_include: VariablesToInclude = None,
           page_size=500,
           bulk_data_as_dataframe: bool = True
           ):
    if session is None:
        session = SofarApi()

    if variables_to_include is None:
        variables_to_include = VariablesToInclude(
            frequencyData=True, waves=True, wind=True, surfaceTemp=True,
            barometerData=True
        )

    start_date_str = datetime_to_iso_time_string(start_date)
    end_date_str = datetime_to_iso_time_string(end_date)

    shape = geometry['type']
    shape_params = geometry['points']
    radius = geometry['radius']

    generator = session.search(
        shape=shape,
        shape_params=shape_params,
        start_date=start_date_str,
        end_date=end_date_str,
        radius=radius,
        page_size=page_size,
        return_generator=True
    )

    spotters = {}
    # loop over all spotters returned
    for spotter in generator:
        spotter_id = spotter['spotterId']
        # loop over keys we can parse
        for key in variables_to_include:
            if key in spotter:
                item = spotter[key]
                if not item:
                    # no data
                    continue

                item['latitude'] = spotter['latitude']
                item['longitude'] = spotter['longitude']
                item['timestamp'] = spotter['timestamp']
                data = _get_class(key, item)
                if spotter_id not in spotters:
                    spotters[spotter_id] = {}

                if key not in spotters[spotter_id]:
                    spotters[spotter_id][key] = []

                spotters[spotter_id][key].append(data)


    for spotter_id in spotters:
        for key in spotters[spotter_id]:
            if bulk_data_as_dataframe and (not key == 'frequencyData'):
                spotters[spotter_id][key]= as_dataframe(spotters[spotter_id][key])
    return spotters


# 6) Helper Functions
# ======================
def _unique_filter(data):
    """
    Filter for dual time entries that occur due to bugs in wavefleet (same
    record returned twice)
    :param data:
    :return:
    """
    timestamps = numpy.array(
        [to_datetime(x['timestamp']).timestamp() for x in data])
    _, unique_indices = numpy.unique(timestamps, return_index=True)

    data = [data[index] for index in unique_indices]

    return data


def _none_filter(data: List[Union[WaveSpectrum1D, WaveBulkData]]):
    F = lambda x: (x['latitude'] is not None) and (
            x['longitude'] is not None) and \
                  (x['timestamp'] is not None)

    return list(filter(F, data))


def _get_class(key, data) -> Union[MetoceanData, WaveSpectrum1D]:
    if key == 'waves':
        return WaveBulkData(
            timestamp=data['timestamp'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            significant_waveheight=data['significantWaveHeight'],
            peak_period=data['peakPeriod'],
            mean_period=data['meanPeriod'],
            peak_direction=data['peakDirection'],
            peak_directional_spread=data['peakDirectionalSpread'],
            mean_direction=data['meanDirection'],
            mean_directional_spread=data['meanDirectionalSpread'],
            peak_frequency=1.0 / data['peakPeriod']
        )
    elif key == 'frequencyData':
        return WaveSpectrum1D(WaveSpectrum1DInput(**data))
    elif key == 'wind':
        return WindData(**data)
    elif key == 'surfaceTemp':
        return SSTData(**data)
    elif key == 'barometerData':
        return BarometricPressure(**data)
    else:
        raise Exception('Unknown variable')
