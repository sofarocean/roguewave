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

from typing import Dict, List, Union, Sequence, Literal, Any
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
    **kwargs,
) -> Dict[str, FrequencySpectrum]:
    """
    Gets the requested frequency wave data for the spotter(s) in the given
    interval.

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a List that for each returned time contains a
    WaveSpectrum1D object.

    """
    return get_spotter_data(
        spotter_ids, "frequencyData", start_date, end_date, **kwargs
    )


def get_bulk_wave_data(
    spotter_ids: Union[str, Sequence[str]],
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    **kwargs,
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

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a dataframe containing the output.
    """
    df = get_spotter_data(spotter_ids, "waves", start_date, end_date, **kwargs)
    return {key: value for (key, value) in df.groupby(["spotter_id"])}


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
    **kwargs,
) -> Dict[str, Dict[str, Union[FrequencySpectrum, DataFrame]]]:
    """
    DEPRICATED USE get_spotter_data instead
    """
    from warnings import warn

    warn(
        "The get_data function is depricated, please use get_spotter_data instead",
        DeprecationWarning,
        stacklevel=2,
    )

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
    data_to_get: List[DATA_TYPES]

    data = {}
    for data_type in data_to_get:
        df = get_spotter_data(spotter_ids, data_type, start_date, end_date, **kwargs)
        if not data_type == "frequencyData":
            df = {key: value for (key, value) in df.groupby(["spotter_id"])}

        for spotter_id in df:
            if spotter_id not in data:
                data[spotter_id] = {}
            data[spotter_id][data_type] = df[spotter_id]

    return data


# Return types for get spotter data. Options that control the return type are:
# - data-to-get-entries; if frequencyData is requested we return a FrequencySpectrum object,
#                        otherwise we return a dataframe.
# - data-to_get type; if a list we return each requested variable as a key value mapping
# - flatten: if false data for each spotter is returned in a key value mapping with
#            spotter_id as key.


def get_spotter_data(
    spotter_ids: Union[str, Sequence[str]],
    data_type: DATA_TYPES,
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    session: SofarApi = None,
    parallel_download=True,
    cache=True,
) -> Union[DataFrame, Dict[str, FrequencySpectrum]]:
    """
    Gets the requested data for the spotter(s) in the given interval as either a dataframe containing
    all the data for the combined spotters in a single table (all datatypes except frequencyData) or
    a dictionary object that has the spotter_id as key and contains a frequency spectrum object as
    values.

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param data_type: Literal string denoting the desired data type, options are
            data_type="waves", bulk wave data
            data_type="wind", wind estimates
            data_type="surfaceTemp", surface temperature (if available)
            data_type="barometerData", barometer data (if available)
            data_type="frequencyData", frequency data (if available) NOTE: does not return a datafrae
            data_type="microphoneData", microphone data if available
            data_type="smartMooringData", smartmooring data if available.

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

    :return:
        data_type="frequencyData": a dictionary with spoter ids as keys and FrequencySpectra as values
        data_type= ...  : a Pandas Dataframe with a spotter_id column that indicates to which spotter entries
            belong.
    """

    if spotter_ids is None:
        spotter_ids = get_spotter_ids()

    if session is None:
        session = _get_sofar_api()

    # Make sure we have a list object
    if not isinstance(spotter_ids, (list, tuple)):
        spotter_ids = [spotter_ids]

    # Ensure we have datetime aware objects
    start_date = to_datetime_utc(start_date)
    end_date = to_datetime_utc(end_date)

    description = f"Downloading from api - {data_type}"
    if not cache or end_date > datetime.now(tz=timezone.utc):
        # Use cached data  _only_ for requests that concern the past. In real-time
        # things may change. In the later case (or when we do not want to use cached values) we flush the cache entries.
        spotter_cache.flush(
            spotter_ids,
            session,
            _download_data,
            var_name=data_type,
            start_date=start_date,
            end_date=end_date,
        )

    data_for_variable = spotter_cache.get_data(
        spotter_ids,
        session,
        _download_data,
        parallel=parallel_download,
        description=description,
        var_name=data_type,
        start_date=start_date,
        end_date=end_date,
    )

    if data_type == "frequencyData":
        data = {}
        for spotter_id, spotter_data in zip(spotter_ids, data_for_variable):
            #
            # Did we get any data for this spotter
            if spotter_data is not None:
                data[spotter_id] = spotter_data
    else:
        values, _id = [], []
        for spotter_id, spotter_data in zip(spotter_ids, data_for_variable):
            #
            # Did we get any data for this spotter
            if spotter_data is not None:
                values += [spotter_data]
                _id += [spotter_id]

        data = concat(values, keys=_id, names=["spotter_id", "time index"])

        data.reset_index(inplace=True)
        data.drop(columns="time index", inplace=True)

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
) -> Union[FrequencySpectrum, Dict[str, Any]]:
    """
    Generator function to unpaginate data from the api.

    idiosyncrasies to handle:
    - Wavefleet sometimes returns multiple instances of the same record (same
      time). These are filtered through a all to _unique. This should
      be fixed in the future.
    - Some entries are broken (None for entries).
    - Not all Spotters will have (all) data for the given timerange.
    - Wavefleet sometimes times out on requests.

    :param spotter_id: Spotter id

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

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
