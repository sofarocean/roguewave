"""
Contents: Routines to get spectral data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to get spectral data from the spotter api

Functions:

- `get_spectrum_from_sofar_spotter_api`, function to call externally do download
                                         spectral data. Parallel if multiple Spotters
                                         are given.
- `_get_next_20_spectra`, Internal helper function. Grabs the next 20 spectra
                          from the given start data. Needed because the Spotter_API
                          will return a maximum of 20 spectra per call.
- '_download_spectra', function that abstracts away the limitation of 20 spectra
                       per call. Handles updating the start date etc.

How To Use This Module
======================
(See the individual functions for details.)

1. Import it: ``import roguewave.externaldata.spotterapi``
2. call ``get_spectrum_from_sofar_spotter_api`` to get the desired data.

TOC
======================

1) Imports: all external import data
2) Constants: all constants used in the module
3) Interfaces: overloaded interfaces for the main function (handles typing, no
               actual logic)
4) Main function: implementation of the main function
    - get_spectrum_from_sofar_spotter_api
5) Private Functions: module private functions that are used by the main function:
    - _get_next_20_spectra
    - _download_spectra
"""

# 1) Imports
#======================
from datetime import datetime, timedelta
from .exceptions import ExceptionNoFrequencyData
from multiprocessing.pool import ThreadPool
from pysofar.spotter import Spotter, SofarApi
from roguewave import logger
from roguewave.tools import datetime_to_iso_time_string, to_datetime
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, \
    WaveSpectrum1DInput
from typing import Dict,List,Union, overload

# 2) Constants
#======================

# Maximum number of spectra to retrieve from the Spotter API per API call. Note
# that 2- os a hard limit of the API. If set higher than 20 it will just return
# 20 (and the implementation will fail)
MAX_LOCAL_LIMIT = 20

# Maximum number of workers in the Threadpool. Should be set to something reasonable
# to not overload wavefleet
MAXIMUM_NUMBER_OF_WORKERS = 40


# 3) Interfaces
#======================
# these overloads only exist so we can make use of typing and autocomplete in
# e.g. Pycharm. Specifically, it handles the case that the return type depends
# on the input.
@overload
def get_spectrum_from_sofar_spotter_api(
        spotter_ids: List[str],
        start_date: Union[datetime, int,float, str] = None,
        end_date: Union[datetime, int,float, str] = None,
        session: SofarApi=None,
        limit=None
) -> Dict[str, List[WaveSpectrum1D]]: ...

@overload
def get_spectrum_from_sofar_spotter_api(
        spotter_ids: str,
        start_date: Union[datetime, int,float, str] = None,
        end_date: Union[datetime, int,float, str] = None,
        session: SofarApi=None,
        limit=None
) -> List[WaveSpectrum1D]: ...

# 4) Main Function
#======================
# Implements the interfaces above.


def get_spectrum_from_sofar_spotter_api(
        spotter_ids: Union[str,List[str]],
        start_date: Union[datetime, int,float,str] = None,
        end_date: Union[datetime, int,float, str] = None,
        session: SofarApi=None,
        parallel_download=True
) -> Union[ Dict[str, List[WaveSpectrum1D]], List[WaveSpectrum1D]]:
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

    if not isinstance( spotter_ids, list ):
        spotter_ids = [spotter_ids]
        return_list = False
    else:
        return_list = True

    if session is None:
        session = SofarApi()


    data = {}
    n = 0
    def worker( spotter_id):
        nonlocal n
        logger.info(f'Downloading data for spotter {spotter_id}')
        try:
            data = _download_spectra(spotter_id,session,start_date,end_date)
        except ExceptionNoFrequencyData as e:
            data = None
        n+=1
        progress = n/len(spotter_ids) * 100
        logger.info( f'spotter: {spotter_id} - progress: {progress:06.2f} %')
        return data

    if parallel_download:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            out = pool.map(worker, spotter_ids)

        for spotter_id, spectra in zip(spotter_ids,out):
            data[spotter_id] = spectra
    else:
        for spotter_id in spotter_ids:
            logger.info( f'Downloading data for spotter {spotter_id}')
            data[spotter_id] = worker(spotter_id)

    if not return_list:
        return data[spotter_ids[0]]
    else:
        return data

# 5) Private Functions
#======================


def _get_next_20_spectra(
        spotter: Spotter,
        start_date: Union[datetime, str,int,float] = None,
        end_date: Union[datetime, str,int,float] = None,
        limit: int = MAX_LOCAL_LIMIT,
) -> List[WaveSpectrum1D]:
    """
    Function that downloads the next 20 Spectra from the Spotter API that lie
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

    json_data = spotter.grab_data(
        limit=limit,
        start_date=datetime_to_iso_time_string(start_date),
        end_date=datetime_to_iso_time_string(end_date),
        include_frequency_data=True,
        include_directional_moments=True)

    if not json_data['frequencyData']:
        raise ExceptionNoFrequencyData(
            f'Spotter {spotter.id} has no spectral data for the requested time range')

    out = []
    for spectrum in json_data['frequencyData']:
        # Load the data into the input object- this is a one-to-one
        # mapping of keys between dictionaries.
        wave_spectrum_input = WaveSpectrum1DInput(**spectrum)
        out.append(WaveSpectrum1D(wave_spectrum_input))

    return out


def _download_spectra(
        spotter_id:str,
        session:SofarApi,
        start_date:Union[datetime, str,int,float] = None,
        end_date:Union[datetime, str,int,float] = None,
        limit:int=None)-> List[WaveSpectrum1D]:
    """
    Function that downloads the Spectra from the API for the requested Spotter
    It abstracts away the limitation that the API can only return a maximum
    of 20 Spectra for a single Spotter per call.

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

    # Set the initial start date. This will get advanced for every 20 spectra
    # we download
    _start_date = start_date
    data = []

    while True:
        # We can only download a maximum of 20 spectra at a time; so we need
        # to loop our request. We do not know how many spotters there are
        # in the given timeframe.
        #
        # Assumptions:
        #   - spotter api returns a maximum of 20 items per requests
        #   - requests returned start from the requested start data and
        #     with the last entry either being the last entry that fits
        #     in the requested window, or merely the last sample that fits
        #     in the 20 items.
        #   - requests returned are in order

        if limit is not None:
            # If we hit the limit (if given) of spotters requested, break
            if len(data) >= limit:
                break

            # otherwise, update our next request so we don't overrun
            # the limit
            local_limit = min(limit - len(data), MAX_LOCAL_LIMIT)
        else:
            # if no limit is given, just ask for the maximum allowed.
            local_limit = MAX_LOCAL_LIMIT

        try:
            # Try to get the next batch of spectra
            next = _get_next_20_spectra(spotter, _start_date,
                                        end_date, local_limit)

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
        data += next

        # If we did not receive all data we requested...
        if len(next) < local_limit:
            # , we are done...
            break
        else:
            # ... else we update the startdate to be the timestamp of the last
            # known entry we received plus a second, and use this as the new
            # start.
            _start_date = to_datetime(next[-1].timestamp) + timedelta(
                seconds=1)
    return data


