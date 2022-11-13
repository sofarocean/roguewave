"""
Contents: Routines to get data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to get data from the spotter search api

Functions:

- `search_circle`, get all available data within a given circle.
- `search_rectangle`, get all available data within a given rectangle.

"""
# 1) Imports
# =============================================================================
from datetime import datetime
from pysofar.spotter import SofarApi
from roguewave.tools.time import datetime_to_iso_time_string
from roguewave.wavespectra import concatenate_spectra

from typing import (
    Union,
    Tuple,
    Literal,
    Sequence,
)
from .helper_functions import _get_sofar_api, _get_class, _unique_filter, as_dataframe
import roguewave.spotterapi.spotter_cache as spotter_cache

# 2) Constants & Private Variables
# =============================================================================


DATA_TYPES = Literal[
    "waves", "wind", "surfaceTemp", "barometerData", "frequencyData", "microphoneData"
]


# 3) Main Functions
# =============================================================================
def search_circle(
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    center_lat_lon: Tuple,
    radius: float,
    session: SofarApi = None,
    cache=True,
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

    geometry = {"type": "circle", "points": center_lat_lon, "radius": radius}

    if session is None:
        session = _get_sofar_api()

    if cache:
        return spotter_cache.get_data_search(
            handler=_search,
            session=session,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry,
        )
    else:
        #
        return _search(start_date, end_date, geometry, session)


def search_rectangle(
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    bounding_box,
    session: SofarApi = None,
    cache=True,
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
    geometry = {"type": "envelope", "points": bounding_box, "radius": None}

    if session is None:
        session = _get_sofar_api()

    print("Get Spotter data: retrieving all data from spatio-temporal region")
    if cache:
        return spotter_cache.get_data_search(
            _search,
            session,
            start_date=start_date,
            end_date=end_date,
            geometry=geometry,
        )
    else:
        #
        return _search(start_date, end_date, geometry, session)


# 5) Private Functions
# =============================================================================
def _search(
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    geometry: dict,
    session: SofarApi = None,
    variables: Sequence[DATA_TYPES] = None,
    page_size=500,
):
    if session is None:
        session = _get_sofar_api()

    if variables is None:
        variables = ["waves", "wind", "surfaceTemp", "barometerData", "frequencyData"]

    start_date_str = datetime_to_iso_time_string(start_date)
    end_date_str = datetime_to_iso_time_string(end_date)

    shape = geometry["type"]
    shape_params = geometry["points"]
    radius = geometry["radius"]

    generator = session.search(
        shape=shape,
        shape_params=shape_params,
        start_date=start_date_str,
        end_date=end_date_str,
        radius=radius,
        page_size=page_size,
        return_generator=True,
    )

    spotters = {}
    # loop over all spotters returned
    for spotter in generator:
        spotter_id = spotter["spotterId"]
        # loop over keys we can parse
        for key in variables:
            #

            if key in spotter:
                item = spotter[key]
                if not item:
                    # no data
                    continue

                item["latitude"] = spotter["latitude"]
                item["longitude"] = spotter["longitude"]
                item["timestamp"] = spotter["timestamp"]
                data = _get_class(key, item)
                if spotter_id not in spotters:
                    spotters[spotter_id] = {}

                if key not in spotters[spotter_id]:
                    spotters[spotter_id][key] = []

                spotters[spotter_id][key].append(data)

    for spotter_id in spotters:
        for key in spotters[spotter_id]:
            # ensure results are unique
            unique = _unique_filter(spotters[spotter_id][key])

            if key == "frequencyData":
                spotters[spotter_id][key] = concatenate_spectra(unique, dim="time")
            else:
                spotters[spotter_id][key] = as_dataframe(unique)

    return spotters
