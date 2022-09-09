"""
Contents: Routines to get data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to get data from the spotter api

Functions:

- `get_spectrum`, function to download spectral data.
- `get_bulk_wave_data`, function to download bulk wave data.
- `get_data`, general function to download different data types (spectral,
    bulk wave, wind, SST, barometer).
- `search_circle`, get all available data within a given circle.
- `search_rectangle`, get all available data within a given rectangle.

"""

# 1) Imports
# =============================================================================
from datetime import datetime, timedelta, timezone
from .exceptions import ExceptionNoDataForVariable
from multiprocessing.pool import ThreadPool
from pysofar.spotter import Spotter, SofarApi
from roguewave import logger
from roguewave.tools.time import datetime_to_iso_time_string, to_datetime_utc
from roguewave.wavespectra  import FrequencySpectrum, create_1d_spectrum

from roguewave.metoceandata import WaveBulkData, as_dataframe, WindData, \
    SSTData, MetoceanData, BarometricPressure
from typing import Dict, List, Union, TypedDict, Tuple, Sequence
from tqdm import tqdm
from pandas import DataFrame
from .helper_functions import _get_sofar_api, get_spotter_ids
import numpy
import roguewave.spotterapi.spotter_cache as spotter_cache
from xarray import Dataset, concat

# 2) Constants & Private Variables
# =============================================================================
# Maximum number of spectra to retrieve from the Spotter API per API call. Note
# that 2- os a hard limit of the API. If set higher than 100 it will just
# return 100 (and the implementation will fail)
MAX_LOCAL_LIMIT = 100
MAX_LOCAL_LIMIT_BULK = 500

# Maximum number of workers in the Threadpool. Should be set to something
# reasonable to not overload wavefleet
MAXIMUM_NUMBER_OF_WORKERS = 40

# Number of retry attemps if a download fails.
NUMBER_OF_RETRIES = 2


# 3) Classes
# =============================================================================
class VariablesToInclude(TypedDict):
    """
    Dictionary that is used to indicate which data to download. The keys
    correspond exactly to the categories returned by a wavefleet request.
    As a consequence we also use the keys as an enumeration of which variables
    are potentially available. This class is internal- should not be
    instantiated or returned.
    """
    frequencyData: bool
    waves: bool
    wind: bool
    surfaceTemp: bool
    barometerData: bool


# 4) Main Functions
# =============================================================================
def get_spectrum(
        spotter_ids: Union[str, Sequence[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download=True,
        cache: bool = True
) -> Dict[str, FrequencySpectrum]:
    """
    Gets the requested frequency wave data for the spotter(s) in the given
    interval

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param session:    Active SofarApi session. If none is provided one will be
                       creatated automatically. This requires that an API key
                       is set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                        retrieving data. Only useful for large requests.

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a List that for each returned time contains a
    WaveSpectrum1D object.

    """
    data = get_data(
        spotter_ids,
        start_date,
        end_date,
        include_frequency_data=True,
        include_waves=False,
        include_wind=False,
        include_surface_temp_data=False,
        session=session,
        parallel_download=parallel_download,
        cache=cache
    )
    out = {}
    for key in data:
        out[key] = data[key]['frequencyData']
    return out


# -----------------------------------------------------------------------------


def get_bulk_wave_data(
        spotter_ids: Union[str, Sequence[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        session: SofarApi = None,
        parallel_download: bool = True,
        bulk_data_as_dataframe: bool = True,
        cache: bool = True,
        convert_to_sofar_model_names = False
) -> Dict[str, Union[DataFrame,list[WaveBulkData]]]:
    """
    Gets the requested bulk wave data for the spotter(s) in the given interval

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param session:    Active SofarApi session. If none is provided one will be
                       creatated automatically. This requires that an API key
                       is set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                        retrieving data. Only useful for large requests.

    :param bulk_data_as_dataframe: return bulk data as a dataframe per Spotter
                       instead of a list of BulkWaveVariable objects.
                       Defaults to yes- only set to false for dev purposes.

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a dataframe containing the output.
    """
    data = get_data(
        spotter_ids,
        start_date,
        end_date,
        include_frequency_data=False,
        include_waves=True,
        include_wind=False,
        include_surface_temp_data=False,
        session=session,
        parallel_download=parallel_download,
        bulk_data_as_dataframe=bulk_data_as_dataframe,
        cache=cache,
        convert_to_sofar_model_names=convert_to_sofar_model_names
    )
    out = {}
    for key in data:
        out[key] = data[key]['waves']
    return out


# -----------------------------------------------------------------------------
def get_data(
        spotter_ids: Union[str, List[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        include_frequency_data=False,
        include_waves=True,
        include_wind=False,
        include_barometer_data=False,
        include_surface_temp_data=False,
        session: SofarApi = None,
        parallel_download=True,
        bulk_data_as_dataframe=True,
        spectral_data_as_dataset_spectrum=True,
        cache=True,
        convert_to_sofar_model_names=False
) -> Dict[str, Dict[str, Union[FrequencySpectrum,
                               list[WaveBulkData], DataFrame]]]:
    """
    Gets the requested data for the spotter(s) in the given interval

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param include_frequency_data: set to True to return spectral data
        (if available)

    :param include_waves: set to True to return wave data (if available)

    :param include_wind: set to True to return wind data (if available)

    :param include_surface_temp_data: set to True to return SST data
        (if available)

    :param include_barometer_data: set to True to return barometer data
        (if available)

    :param session:    Active SofarApi session. If none is provided one will be
                       created automatically. This requires that an API key is
                       set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                       retrieving data. Only useful for large requests.

    :param bulk_data_as_dataframe: return bulk data as a dataframe per Spotter
                       instead of a list of BulkWaveVariable objects. Defaults
                       to yes- only set to false for dev purposes.

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a dataframe containing the output.

    To Note; this function now fetches each of the variables seperately -
        instead of doing one fetch; this is a clutch fix to the issue that
        data for different variables have different timestamps. As a consequence
        getting 100 data points for frequency data may cover 5 days -whereas
        the same number of points only cover a day for SST data. Since we
        only get 100 data points per request- and have to advance to the next
        date ourselves-  this complicates matters tremendously.
    """

    variables_to_include = VariablesToInclude(
        waves=include_waves,
        wind=include_wind,
        surfaceTemp=include_surface_temp_data,
        frequencyData=include_frequency_data,
        barometerData=include_barometer_data,
    )
    out = {}
    for variable,to_include in variables_to_include.items():
        if to_include:
            print(
                f"Getting spotter data: retrieving {variable}")
            dummy = VariablesToInclude(
                frequencyData=False,
                wind=False,
                waves=False,
                surfaceTemp=False,
                barometerData=False)
            dummy[variable] = to_include

            temp = _get_data(
                spotter_ids,
                start_date,
                end_date,
                dummy['frequencyData'],
                dummy['waves'],
                dummy['wind'],
                dummy['barometerData'],
                dummy['surfaceTemp'],
                session,
                parallel_download,
                bulk_data_as_dataframe,
                spectral_data_as_dataset_spectrum,
                cache,
                convert_to_sofar_model_names
            )
            for spotter_id in temp:
                if spotter_id not in out:
                    out[spotter_id] = {}
                out[spotter_id][variable] = temp[spotter_id][variable]
    return out

def _get_data(
        spotter_ids: Union[str, List[str]],
        start_date: Union[datetime, int, float, str] = None,
        end_date: Union[datetime, int, float, str] = None,
        include_frequency_data=False,
        include_waves=True,
        include_wind=False,
        include_barometer_data=False,
        include_surface_temp_data=False,
        session: SofarApi = None,
        parallel_download=True,
        bulk_data_as_dataframe=True,
        spectral_data_as_dataset_spectrum=True,
        cache=True,
        convert_to_sofar_model_names=False
) -> Dict[str, Dict[str, Union[FrequencySpectrum,
                               list[WaveBulkData], DataFrame]]]:
    """
    Gets the requested data for the spotter(s) in the given interval

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param include_frequency_data: set to True to return spectral data
        (if available)

    :param include_waves: set to True to return wave data (if available)

    :param include_wind: set to True to return wind data (if available)

    :param include_surface_temp_data: set to True to return SST data
        (if available)

    :param include_barometer_data: set to True to return barometer data
        (if available)

    :param session:    Active SofarApi session. If none is provided one will be
                       created automatically. This requires that an API key is
                       set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                       retrieving data. Only useful for large requests.

    :param bulk_data_as_dataframe: return bulk data as a dataframe per Spotter
                       instead of a list of BulkWaveVariable objects. Defaults
                       to yes- only set to false for dev purposes.

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a dataframe containing the output.
    """

    if spotter_ids is None:
        spotter_ids = get_spotter_ids()

    variables_to_include = VariablesToInclude(
        waves=include_waves,
        wind=include_wind,
        surfaceTemp=include_surface_temp_data,
        frequencyData=include_frequency_data,
        barometerData=include_barometer_data,
    )

    # Make sure we have a list object
    if not isinstance(spotter_ids, list):
        spotter_ids = [spotter_ids]

    if cache and end_date < datetime.now(tz=timezone.utc):
        # Use the cache _only_ for requests that concern the past. In real-time
        # things may change.

        out = spotter_cache.get_data(
            spotter_ids,
            session,
            _download_data,
            variables_to_include=variables_to_include,
            start_date=start_date,
            end_date=end_date,
            bulk_data_as_dataframe=bulk_data_as_dataframe,
            convert_to_sofar_model_names=convert_to_sofar_model_names)
    else:
        if session is None:
            session = _get_sofar_api()

        def worker(_spotter_id):
            return _download_data(_spotter_id, session, variables_to_include,
                                  start_date, end_date, bulk_data_as_dataframe,
                                  limit=None,
                                  convert_to_sofar_model_names=
                                  convert_to_sofar_model_names)

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
            if 'frequencyData' in spotter_data:
                if isinstance(spotter_data['frequencyData'],List):
                    # This will be deprecated in a future version.
                    spotter_data['frequencyData'] = \
                        FrequencySpectrum.concat_from_list(
                            spotter_data['frequencyData'])
            data[spotter_id] = spotter_data

    return data


# -----------------------------------------------------------------------------


def search_circle(
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        center_lat_lon: Tuple,
        radius: float,
        session: SofarApi = None,
        cache=False
):
    """
    Search for all Spotters that have data available within the give spatio-
    temporal region defined by a circle with given center and radius and start-
    and end- dates. This calls the "search" endpoint of wavefleet.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
    :param center_lat_lon: (Latitude, Longitude) of the center of the circle.
    :param radius: Radius in meter of the circle.
    :param session: Active SofarApi session. If none is provided one will be
                    created automatically. This requires that an API key is
                    set in the environment.
    :return:
    """

    geometry = {'type': 'circle', 'points': center_lat_lon, 'radius': radius}

    if session is None:
        session = _get_sofar_api()

    print("Get Spotter data: retrieving all data from spatio-temporal region")
    if cache:
        return spotter_cache.get_data_search(
            handler=_search,
            session=session,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry
        )
    else:
        #
        return _search(start_date, end_date, geometry, session)


# -----------------------------------------------------------------------------


def search_rectangle(
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        bounding_box,
        session: SofarApi = None,
        cache=False
):
    """
    Search for all Spotters that have data available within the give spatio-
    temporal region defined by a circle with given center and radius and start-
    and end-dates. This calls the "search" endpoint of wavefleet.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
    :param bounding_box: coordinates of two points that define a rectangular
    bounding box. Coordinates per point are given as (lat, lon) pairs, and the
    input takes the form of a list/tuple of points: ( (p1_lat, p1_lon),(p2_lat,
    p2_lon) )
    :param session: Active SofarApi session. If none is provided one will be
                    created automatically. This requires that an API key is
                    set in the environment.
    :return:
    """
    geometry = {'type': 'envelope',
                'points': bounding_box,
                'radius': None}

    if session is None:
        session = _get_sofar_api()

    print("Get Spotter data: retrieving all data from spatio-temporal region")
    if cache:
        return spotter_cache.get_data_search(
            _search,
            session,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry
        )
    else:
        #
        return _search(start_date, end_date, geometry, session)


# -----------------------------------------------------------------------------


# 5) Private Functions
# =============================================================================
def _download_data(
        spotter_id: str,
        session: SofarApi,
        variables_to_include: VariablesToInclude,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        bulk_data_as_dataframe: bool = True,
        limit: int = None,
        convert_to_sofar_model_names=False) -> \
        Dict[str, Union[List[FrequencySpectrum], List[WaveBulkData]]]:
    """
    Function that downloads data from the API for the requested Spotter
    It abstracts away the limitation that the API can only return a maximum
    of 100 Spectra or 500 bulk data points for a single Spotter per call.

    :param spotter_id: ID for the Spotter we want to download
    :param session: Active session of the SofarAPI
    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history
    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history
    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to
                  change it. (calling function does not set the limit)

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
            _next, number_of_items_returned, max_time = \
                _get_next_page(spotter, variables_to_include,
                               _start_date, end_date, local_limit)

        except ExceptionNoDataForVariable as e:
            # Could not download data, add nothing, raise warning
            raise e

        # Add the data to the list

        for key in _next:
            if key in data:
                data[key] += _next[key]
            else:
                data[key] = _next[key]

        #
        # HACK WARNING: currently the API returns exactly half the number of
        # samples requested for surfaceTemp. So we need to half the checking
        # limit as well.
        if variables_to_include['surfaceTemp']:
            actual_limit = local_limit/2
        else:
            actual_limit = local_limit

        # If we did not receive all data we requested...
        if number_of_items_returned < actual_limit:
            # , we are done...
            break
        else:
            # ... else we update the startdate to be the time of the last
            # known entry we received plus a second, and use this as the new
            # start.
            _start_date = to_datetime_utc(max_time,to_scalar=True) + timedelta(seconds=1)

    # Postprocessing
    if len(data) < 1:
        data = None
    else:
        if bulk_data_as_dataframe:
            # Convert bulk data to a dataframe if desired
            for key in data:
                if key == 'frequencyData':
                    # Concatenate all the spectral dataset along time dimension
                    # and return a wave frequency spectrum object
                    data[key] = FrequencySpectrum.concat_from_list(data[key])
                else:
                    data[key] = as_dataframe(
                        data[key],convert_to_sofar_model_names)
    return data


# -----------------------------------------------------------------------------


def _get_next_page(
        spotter: Spotter,
        variables_to_include: VariablesToInclude,
        start_date: Union[datetime, str, int, float] = None,
        end_date: Union[datetime, str, int, float] = None,
        limit: int = MAX_LOCAL_LIMIT,
) -> Tuple[Dict[str, Union[List[Dataset],
                           List[WaveBulkData]]], int, datetime]:
    """
    Function that downloads the page of Data from the Spotter API that lie
    within the given interval, starting from the record closest to the
    startdate.

    idiosyncrasies to handle:
    - Wavefleet sometimes returns multiple instances of the same record (same
      time). These are filtered through a all to _unique. This should
      be fixed in the future.
    - Some entries are broken (None for entries).
    - Not all Spotters will have (all) data for the given timerange.
    - Wavefleet sometimes times out on requests.

    :param spotter: Spotter object from pysofar

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :param limit: Maximum number of Spectra to download per call. Not exposed
                  externally right now as there is not really a reason to
                  change it. (calling function does not set the limit)

    :return: Data
    """

    # Retry mechanism
    start_date = to_datetime_utc(start_date)
    end_date = to_datetime_utc(end_date)
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
                warning = f'Error downloading data for {spotter.id}, ' \
                          f'attempting retry {retry + 1}'
                logger.warning(warning)
            else:
                raise e

    out = {}
    max_num_items = 0
    max_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
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

            # If so- were any elements returned for this period? If not
            # continue. We could error here, but there may be gaps in data for
            # certain sensors, so we try to continue
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

            # Add to output
            out[var_name] = \
                [_get_class(var_name, data) for data in json_data[var_name]]

            # Filter for doubles.
            out[var_name] = _unique_filter(out[var_name])

            # get the new max time. Our object can be a FrequencySpectrum or
            # a metocean data object here. The latter returns a scalar date-
            # time object, the former returns a length 1 data-array of
            # datetime64. To ensure we get a datetime - we call the datetime
            # conversion and request output as a scalar.
            if to_datetime_utc(out[var_name][-1].time,True) > max_time:
                max_time = to_datetime_utc(out[var_name][-1].time, True)

    return out, max_num_items, max_time


# -----------------------------------------------------------------------------


def _search(start_date: Union[datetime, str],
            end_date: Union[datetime, str],
            geometry: dict,
            session: SofarApi = None,
            variables_to_include: VariablesToInclude = None,
            page_size=500,
            bulk_data_as_dataframe: bool = True,
            ):
    if session is None:
        session = _get_sofar_api()

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
        for key, value in variables_to_include.items():
            #
            if not value:
                # Do we want to include this variable?
                continue

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
            # ensure results are unique
            spotters[spotter_id][key] = _unique_filter(
                spotters[spotter_id][key])

            if bulk_data_as_dataframe and (not key == 'frequencyData'):
                spotters[spotter_id][key] = as_dataframe(
                    spotters[spotter_id][key])

            if key == 'frequencyData':
                spotters[spotter_id][key] = FrequencySpectrum.concat_from_list(
                    spotters[spotter_id][key]
                )
    return spotters


# -----------------------------------------------------------------------------


# 6) Helper Functions
# =============================================================================
def _unique_filter(data):
    """
    Filter for dual time entries that occur due to bugs in wavefleet (same
    record returned twice)
    :param data:
    :return:
    """

    # Get time
    if isinstance(data[0],FrequencySpectrum ):
        time = numpy.array(
            [to_datetime_utc(x['time'].values[0]).timestamp() for x in data])
    else:
        time= numpy.array([x.time.timestamp() for x in data])

    # Get indices of unique times
    _, unique_indices = numpy.unique(time, return_index=True)

    # Return only unique indices
    return [data[index] for index in unique_indices]


# -----------------------------------------------------------------------------


def _none_filter(data: Dict):
    """
    Filter for the occasional occurance of bad data returned from wavefleet.
    :param data:
    :return:
    """
    return list(
        filter(
            lambda x: (x['latitude'] is not None) and (
                    x['longitude'] is not None) and (
                                  x['timestamp'] is not None),
            data
        )
    )


# -----------------------------------------------------------------------------


def _get_class(key, data) -> Union[MetoceanData, FrequencySpectrum]:
    if key == 'waves':
        return WaveBulkData(
            time=data['timestamp'],
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
        return create_1d_spectrum(
            frequency=numpy.array(data['frequency']),
            variance_density=numpy.array(data['varianceDensity']),
            time=to_datetime_utc(data['timestamp']),
            latitude=numpy.array(data['latitude']),
            longitude=numpy.array(data['longitude']),
            a1=numpy.array(data['a1']),
            b1=numpy.array(data['b1']),
            a2=numpy.array(data['a2']),
            b2=numpy.array(data['b2'])
            )
    elif key == 'wind':
        data = data.copy()
        data['time'] = data['timestamp']
        data.pop('timestamp')
        return WindData(**data)
    elif key == 'surfaceTemp':
        data = data.copy()
        data['time'] = data['timestamp']
        data.pop('timestamp')
        return SSTData(**data)
    elif key == 'barometerData':
        data = data.copy()
        data['time'] = data['timestamp']
        data.pop('timestamp')
        return BarometricPressure(**data)
    else:
        raise Exception('Unknown variable')
# -----------------------------------------------------------------------------
