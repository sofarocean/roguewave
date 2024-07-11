"""
Contents: Simple file caching routines that automatically cache remote files
          locally for use.

Copyright (C) 2023
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Functions:
- `colocate_points`, function to colocate two point data sources.

"""
from pandas import DataFrame
from scipy.spatial import KDTree
from datetime import timedelta, datetime, timezone
from numpy import cos, sin, pi, empty, nan, unwrap, mean, dtype, full


def colocate_points(
    data: DataFrame, data_to_colocate: DataFrame, radius: float, time_delta: timedelta
) -> DataFrame:
    """
    Collocate data points in the "data_to_colocate" dataframe to points in the original data frame. Both dataframes are
    expected to contain columns denoted: "longitude","latitude","time" filled with corresponding data. Longitude is
    assumed to be in [-180,180), latitude in [-90,90], time is given as a pandas timestamp with or without timezone
    information.

    The functions returns a dataframe of all data in data_to_colocate that is within the given radius and time_delta of
    datapoints in frame 1. The dataframe index corresponds to the point that was matched in "data" dataframe. If
    multiple points in the "data_to_colocate" fall within the requested bounds, the average of those points is returned
    as the result for datatypes for which average is meaningful. Otherwise data from the first point found is returned.

    :param data: reference data
    :param data_to_colocate: data points we want to match to points in the reference data set
    :param radius: maximum spatial radius (in meters) of points we consider a match.
    :param time_delta: maximum time interval of points we consider a match.
    :return:
    """

    # Create the KDTrees
    tree_data = _construct_tree(data, radius, time_delta)
    tree_to_colocate = _construct_tree(data_to_colocate, radius, time_delta)

    # Setup the output structure
    n = len(data["latitude"])
    output = DataFrame()
    for key in data_to_colocate.columns:
        output[key] = full(n, nan, dtype=data_to_colocate[key].dtype)

    # Query the data tree and return all indices of data close in the other tree.
    nearest_neighbour_indices = tree_data.query_ball_tree(tree_to_colocate, radius)

    for index_a, indices_b in enumerate(nearest_neighbour_indices):
        # any matched?, if not continue (leave data set to nan).

        if len(indices_b) > 0:
            for key in data_to_colocate.columns:
                data_entry = data_to_colocate[key].iloc[indices_b]

                # There may be multiple matched. We only assing a single value as matched.
                # Depending on the data type we will return an average or something else.
                if key == "longitude":
                    data_entry = (
                        mean(unwrap(data_entry.values, discont=180, period=360)) + 180
                    ) % 360 - 180

                elif key == "time":
                    data_entry = data_entry.iloc[0]

                elif data_entry.dtype == dtype("O"):
                    data_entry = data_entry.iloc[0]

                else:
                    data_entry = mean(data_entry.values)

                output.at[index_a, key] = data_entry

    # Drop nan entries (no matches) and return dataframe
    return output.dropna(how="all")


def _construct_tree(dataset: DataFrame, radius: float, time_delta: timedelta) -> KDTree:
    """
    Construct a KDTRee based on input data frame. The dataframe must have: latitude, longitude, time as columns. Time
    must be given as a proper datetime object.

    To allow for queries on the tree we convert time to a spatial dimension. To this end we have to couple a spatial
    delta (or distance) to an equivalent time delta, which will be very situational dependent.

    :param dataset: pandas Dataset to load into KDTree
    :param radius: spatial radius for which we want to consider neighbours
    :param time_delta: time delta for which we want to consider neighbours
    :return: scipy KDTree object
    """

    data = empty((4, dataset.shape[0]))

    # Ensure the datetime is timezone aware
    if dataset["time"].dt.tz is not None:
        time = dataset["time"].dt.tz_convert(timezone.utc)

    else:
        time = dataset["time"].dt.tz_localize(timezone.utc)

    # Convert latitude and longitude to radians
    lat = dataset["latitude"] * pi / 180
    lon = dataset["longitude"] * pi / 180

    # Convert lat, lon, time into a 4d spatial tuple.

    # To avoid wrapping complications, we convert the spherical latitude and longitude back to an x,y,z coordinate frame
    # usig WGS84 radius. Note that exact representation is not super critical (spherical vs oblate) - under the
    # assumption that some inaccuracy in nearest neighbour queries based on distance is fine.
    _RADIUS_EARTH = 6371008.8
    data[0, :] = _RADIUS_EARTH * cos(lon) * cos(lat)
    data[1, :] = _RADIUS_EARTH * sin(lon) * cos(lat)
    data[2, :] = _RADIUS_EARTH * sin(lat)

    # Convert time to a spatial dimension using a characteristic velocity. We assume we know a characteristic time
    # and spatial interval within which we consider two points to be matched. To convert time to space we associate with
    # these intervals a characteristic speed.
    time_delta_seconds = (
        time - datetime(1970, 1, 1, tzinfo=timezone.utc)
    ).dt.total_seconds()
    characteristic_speed = radius / time_delta.total_seconds()
    data[3, :] = characteristic_speed * time_delta_seconds

    # Build tree and return
    return KDTree(data.transpose())
