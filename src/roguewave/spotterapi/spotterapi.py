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
from datetime import datetime, timedelta
from .exceptions import ExceptionNoFrequencyData, ExceptionCouldNotDownloadData
from multiprocessing.pool import ThreadPool
from pysofar.spotter import Spotter, SofarApi
from roguewave import logger
from roguewave.tools import datetime_to_iso_time_string, to_datetime
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, \
    WaveSpectrum1DInput
from roguewave.wavespectra.wavespectrum import WaveBulkData
from typing import Dict, List, Union, overload, TypedDict, Tuple
from tqdm import tqdm

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

class VariablesToInclude(TypedDict):
    include_frequency_data: bool
    include_directional_moments: bool
    include_waves: bool
    include_wind: bool
    include_surface_temp_data: bool


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

def get_spotter_ids(sofar_api:SofarApi=None)->List[str]:
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
            parallel_download=parallel_download
    )
    out = {}
    for key in data:
        out[key]=data[key]['spectra']
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
        out[key]=data[key]['waves']
    return out

def get_data(
            spotter_ids: Union[str, List[str]],
            start_date: Union[datetime, int, float, str] = None,
            end_date: Union[datetime, int, float, str] = None,
            include_frequency_data=False,
            include_directional_moments=False,
            include_waves=True,
            include_wind=False,
            include_surface_temp_data=False,
            session: SofarApi = None,
            parallel_download=True
    ) -> Dict[str, Dict[str,Union[list[WaveSpectrum1D],list[WaveBulkData]]]]:

    if spotter_ids is None:
        spotter_ids = get_spotter_ids()

    variables_to_include = VariablesToInclude(
        include_waves=include_waves,
        include_wind=include_wind,
        include_surface_temp_data=include_surface_temp_data,
        include_frequency_data=include_frequency_data,
        include_directional_moments=include_directional_moments
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
                                  start_date, end_date)
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
            data[spotter_id] = {}
            #
            # if so, did we get data for all variables
            for key in spotter_data:
                if len(spotter_data[key]) > 0:
                    data[spotter_id][key] = spotter_data[key]

    if not return_list:
        return data[spotter_ids[0]]
    else:
        return data


# 5) Private Functions
# ======================
def _get_next_100_spectra(
        spotter: Spotter,
        variables_to_include,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        limit: int = MAX_LOCAL_LIMIT,
) -> Tuple[Dict[str, Union[List[WaveSpectrum1D], List[WaveBulkData]]], int]:
    """
    Function that downloads the next 100 Spectra from the Spotter API that lie
    within the given interval, starting from Spectra closest to the startdate.

    :param spotter: Spotter object from pysofar

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history

    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to change
                  it. (calling function does not set the limit)

    :return: Data as a FrequencyDataList Object
    """
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)
    for retry in range(0, NUMBER_OF_RETRIES + 1):
        try:
            json_data = spotter.grab_data(
                limit=limit,
                start_date=datetime_to_iso_time_string(start_date),
                end_date=datetime_to_iso_time_string(end_date),
                include_frequency_data=variables_to_include[
                    'include_frequency_data'],
                include_directional_moments=variables_to_include[
                    'include_directional_moments'],
                include_waves=variables_to_include['include_waves'],
                include_wind=variables_to_include['include_wind'],
                include_surface_temp_data=variables_to_include[
                    'include_surface_temp_data'],
            )
            break
        except Exception as e:
            if retry < NUMBER_OF_RETRIES:
                warning = f'Error downloading data for {spotter.id}, attempting retry {retry + 1}'
                logger.warning(warning)
            else:
                logger.debug(e)
    else:
        warning = f'Error downloading data for {spotter.id} failed for {start_date} to {end_date} \n error logged.'
        logger.warning(warning)
        raise ExceptionCouldNotDownloadData()

    out = {}
    number_of_items_returned = 0
    if variables_to_include['include_frequency_data']:
        if not json_data['frequencyData']:
            out_spectrum = None
        else:
            out_spectrum = []
            number_of_items_returned = len(json_data['frequencyData'])
            for spectrum in json_data['frequencyData']:
                # Load the data into the input object- this is a one-to-one
                # mapping of keys between dictionaries.
                if (spectrum['latitude'] is None) or (
                        spectrum['longitude'] is None):
                    info = f"{spotter.name} at {spotter.timestamp} has None for latitude or longitude. Data is dropped for this time"
                    logger.info(info)
                    out_spectrum.append(None)
                else:
                    wave_spectrum_input = WaveSpectrum1DInput(**spectrum)
                    out_spectrum.append(WaveSpectrum1D(wave_spectrum_input))
        out['spectra'] = out_spectrum

    if variables_to_include['include_waves']:
        if json_data['waves']:
            out_waves = []
            number_of_items_returned = max(number_of_items_returned,
                                           len(json_data['waves']))
            for data in json_data['waves']:
                data: ApiWaveData
                out_waves.append(
                    WaveBulkData(
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
                )
        else:
            out_waves = None
        out['waves'] = out_waves
    return out, number_of_items_returned


def _download_data(
        spotter_id: str,
        session: SofarApi,
        variables_to_include,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        limit: int = None) -> Dict[
    str, Union[List[WaveSpectrum1D], List[WaveBulkData]]]:
    """
    Function that downloads the Spectra from the API for the requested Spotter
    It abstracts away the limitation that the API can only return a maximum
    of 100 Spectra for a single Spotter per call.

    :param spotter_id: ID for the Spotter we want to download
    :param session: Active session of the SofarAPI
    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history
    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to change
                  it. (calling function does not set the limit)

    :return: List of available wavespectra in the requested timeframe. The
    function returns a List of wavespectrum1D objects.
    """

    # Create a Spotter object to Query
    spotter = Spotter(spotter_id, spotter_id, session=session)

    # Set the initial start date. This will get advanced for every 100 spectra
    # we download
    _start_date = start_date

    if variables_to_include['include_frequency_data']:
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
        #   - spotter api returns a maximum of 100 items per requests
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
            # Try to get the next batch of spectra
            next, number_of_items_returned = \
                _get_next_100_spectra(spotter, variables_to_include,
                                      _start_date, end_date, local_limit)
            for key in next:
                if next[key] is not None:
                    next[key] = list(filter(lambda x: x is not None, next[key]))
        except ExceptionCouldNotDownloadData as e:
            # Could not download data, add nothing
            next = None
        except ExceptionNoFrequencyData as e:
            # If no frequency data was returned, we either...
            if not len(data):
                # ...raise an error, if no data was returned previously (no
                # data available at all)...
                raise e
            else:
                # ...or return, assuming that we exhausted the data was
                # available.
                break

        # Add the data to the list
        if next is not None:
            for key in next:
                if next[key] is not None:
                    if len(next[key]) > 0:
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
            _start_date = to_datetime(next[-1].timestamp) + timedelta(
                seconds=1)


    if len(data) < 1:
        data = None
    return data
