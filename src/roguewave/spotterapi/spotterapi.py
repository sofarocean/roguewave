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
from pysofar.spotter import Spotter, SofarApi
from pysofar.wavefleet_exceptions import QueryError
from roguewave import logger
from roguewave.tools.time import (
    datetime_to_iso_time_string,
    to_datetime_utc,
)
from roguewave.wavespectra import (
    FrequencySpectrum,
    concatenate_spectra,
)

from typing import (
    Dict,
    List,
    Union,
    Sequence,
    Literal,
)
from pandas import DataFrame, concat
from .helper_functions import (
    _get_sofar_api,
    get_spotter_ids,
    _unique_filter,
    as_dataframe,
    _none_filter,
    _get_class,
)
import roguewave.spotterapi.spotter_cache as spotter_cache

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

MAX_DAYS_SMARTMOORING = 10

DATA_TYPES = Literal[
    "waves",
    "wind",
    "surfaceTemp",
    "barometerData",
    "frequencyData",
    "microphoneData",
    "smartMooringData",
]


# 4) Main Functions
# =============================================================================


def get_spectrum(
    spotter_ids: Union[str, Sequence[str]],
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    session: SofarApi = None,
    parallel_download=True,
    cache: bool = True,
    flatten=False,
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
    varname: Literal["frequencyData"] = "frequencyData"
    data = get_spotter_data(
        spotter_ids,
        [varname],
        start_date,
        end_date,
        session=session,
        parallel_download=parallel_download,
        cache=cache,
        flatten=flatten,
    )
    return {key: data[key]["frequencyData"] for key in data}


def get_bulk_wave_data(
    spotter_ids: Union[str, Sequence[str]],
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    session: SofarApi = None,
    parallel_download: bool = True,
    cache: bool = True,
    flatten=False,
) -> Dict[str, DataFrame]:
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

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a dataframe containing the output.
    """

    varname: Literal["waves"] = "waves"
    data = get_spotter_data(
        spotter_ids,
        [varname],
        start_date,
        end_date,
        session=session,
        parallel_download=parallel_download,
        cache=cache,
        flatten=flatten,
    )
    return {key: data[key]["waves"] for key in data}


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
    flatten=False,
    session: SofarApi = None,
    parallel_download=True,
    cache=True,
) -> Dict[str, Dict[str, Union[FrequencySpectrum, DataFrame]]]:
    """
    DEPRICATED USE get_spotter_data instead
    """
    data_to_get = []
    if include_frequency_data:
        data_to_get.append("frequencyData")
    if include_barometer_data:
        data_to_get.append("barometerData")
    if include_surface_temp_data:
        data_to_get.append("surfaceTemp")
    if include_wind:
        data_to_get.append("wind")
    if include_waves:
        data_to_get.append("waves")

    return get_spotter_data(
        spotter_ids,
        data_to_get,
        start_date,
        end_date,
        session,
        parallel_download,
        flatten=flatten,
        cache=cache,
    )


def get_spotter_data(
    spotter_ids: Union[str, List[str]],
    data_to_get: Union[DATA_TYPES, List[DATA_TYPES]],
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    flatten=True,
    session: SofarApi = None,
    parallel_download=True,
    cache=True,
) -> Dict[str, Dict[str, Union[FrequencySpectrum, DataFrame]]]:
    """
    Gets the requested data for the spotter(s) in the given interval

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param session:    Active SofarApi session. If none is provided one will be
                       created automatically. This requires that an API key is
                       set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                       retrieving data. Only useful for large requests.

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

    if spotter_ids is None:
        spotter_ids = get_spotter_ids()

    if session is None:
        session = _get_sofar_api()

    # Make sure we have a list object
    if not isinstance(spotter_ids, (list, tuple)):
        spotter_ids = [spotter_ids]

    # Make sure we have a list object
    if not isinstance(data_to_get, (list, tuple)):
        data_to_get = [data_to_get]

    # Ensure we have datetime aware objects
    start_date = to_datetime_utc(start_date)
    end_date = to_datetime_utc(end_date)

    data = {}
    for var_name in data_to_get:
        description = f"Downloading from api - {var_name}"
        if not cache or end_date > datetime.now(tz=timezone.utc):
            # Use cached data  _only_ for requests that concern the past. In real-time
            # things may change. In the later case (or when we do not want to use cached values) we flush the cache entries.
            spotter_cache.flush(
                spotter_ids,
                session,
                _download_data,
                var_name=var_name,
                start_date=start_date,
                end_date=end_date,
            )

        data_for_variable = spotter_cache.get_data(
            spotter_ids,
            session,
            _download_data,
            parallel=parallel_download,
            description=description,
            var_name=var_name,
            start_date=start_date,
            end_date=end_date,
        )

        if not flatten:
            for spotter_id, spotter_data in zip(spotter_ids, data_for_variable):
                #
                # Did we get any data for this spotter
                if spotter_data is not None:
                    if spotter_id not in data:
                        data[spotter_id] = {}
                    data[spotter_id][var_name] = spotter_data
        else:
            values, _id = [], []
            for spotter_id, spotter_data in zip(spotter_ids, data_for_variable):
                #
                # Did we get any data for this spotter
                if spotter_data is not None:
                    values += [spotter_data]
                    _id += [spotter_id]

            if var_name == "frequencyData":
                data[var_name] = concatenate_spectra(values)
            else:
                data[var_name] = concat(
                    values, keys=_id, names=["spotter_id", "time index"]
                )

                data[var_name].reset_index(inplace=True)
                data[var_name] = data[var_name].drop(columns="time index")

    return data


def _download_data(**kwargs) -> Union[FrequencySpectrum, DataFrame]:
    """ """
    data = list(_unpaginate(**kwargs))

    # Postprocessing
    if len(data) < 1:
        data = None
    else:
        if kwargs["var_name"] == "frequencyData":
            data = _unique_filter(data)
            data = concatenate_spectra(data, dim="time")
        else:
            data = as_dataframe(data)
            data.drop_duplicates(inplace=True)
            data.sort_values("time", inplace=True)
    return data


def get_smart_mooring_data(
    start_date: str, end_date: str, spotter: Spotter, max_days=None
):

    # We have a self imposed limit that we return to avoid overloading wavefleet.
    if max_days is not None:
        start_date_dt = to_datetime_utc(start_date)
        end_date_dt = to_datetime_utc(end_date)

        # If the enddate exceeds the maximum number of days to download we reduce the enddate
        if end_date_dt - start_date_dt > timedelta(days=max_days):
            end_date_dt = start_date_dt + timedelta(days=max_days)

            # update the enddate
            end_date = datetime_to_iso_time_string(end_date_dt)

    params = {"spotterId": spotter.id, "startDate": start_date, "endDate": end_date}
    scode, data = spotter._session._get("sensor-data", params=params)

    if scode != 200:
        raise QueryError(data["message"])

    return data["data"]


def _unpaginate(
    var_name: str,
    start_date: datetime,
    end_date: datetime,
    spotter_id: str,
    session: SofarApi,
):
    """
    Generator function to unpaginate data from the api.

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
    spotter = Spotter(spotter_id, spotter_id, session=session)
    while True:
        json_data = None
        for retry in range(0, NUMBER_OF_RETRIES + 1):
            try:
                if var_name == "smartMooringData":
                    json_data = get_smart_mooring_data(
                        start_date=datetime_to_iso_time_string(start_date),
                        end_date=datetime_to_iso_time_string(end_date),
                        spotter=spotter,
                        max_days=MAX_DAYS_SMARTMOORING,
                    )
                    json_data = {"smartMooringData": json_data}
                else:
                    json_data = spotter.grab_data(
                        limit=MAX_LOCAL_LIMIT,
                        start_date=datetime_to_iso_time_string(start_date),
                        end_date=datetime_to_iso_time_string(end_date),
                        include_frequency_data=var_name == "frequencyData",
                        include_directional_moments=var_name == "frequencyData",
                        include_barometer_data=var_name == "barometerData",
                        include_waves=var_name == "waves",
                        include_wind=var_name == "wind",
                        include_surface_temp_data=var_name == "surfaceTemp",
                        include_microphone_data=var_name == "microphoneData",
                    )
                break
            except Exception as e:
                if retry < NUMBER_OF_RETRIES:
                    warning = (
                        f"Error downloading data for {spotter.id}, "
                        f"attempting retry {retry + 1}"
                    )
                    logger.warning(warning)
                else:
                    raise e

        #
        # If so- was it returned? If not- raise error
        if var_name not in json_data:
            raise ExceptionNoDataForVariable(var_name)

        # If no data - return
        if not json_data[var_name]:
            break

        # Filter for Nones.
        json_data[var_name] = _none_filter(json_data[var_name])
        if not json_data[var_name]:
            break

        objects = []
        for _object in [_get_class(var_name, data) for data in json_data[var_name]]:
            if isinstance(_object, FrequencySpectrum):
                date = to_datetime_utc(_object.time.values)
            else:
                date = _object["time"]

            if date < start_date:
                continue
            else:
                objects.append(_object)

        if len(objects) == 0:
            break

        last_object = None
        for _object in objects:
            last_object = _object
            yield _object

        if isinstance(last_object, FrequencySpectrum):
            start_date = to_datetime_utc(last_object.time.values) + timedelta(seconds=1)
        else:
            start_date = last_object["time"] + timedelta(seconds=1)
