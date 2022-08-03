import numpy
from datetime import datetime
from numpy import interp, ndarray
from typing import List,Dict,Union,Tuple, overload, Hashable
from xarray import Dataset, DataArray


def interpolate_dataset( dataset:DataArray, points:Tuple[ndarray,...] ):

    # Get dimension
    dimensions = dataset.dims # type: Tuple[Hashable,...]

    if len(points) != len(dimensions):
        dim_str = [str(dim) for dim in dimensions]
        raise ValueError( f' Points expected to have {len(dimensions)} '
                    f'coordinates corresponding to: { ", ".join(dim_str)}' )


    pass





def interpolation_weights(latitude,
                          longitude,
                          model_latitudes,
                          model_longitudes):

    points = enclosing_points(latitude, longitude, model_latitudes, model_longitudes)

    factors = []

    latitude_delta = model_latitudes[points[2][0]] - model_latitudes[points[0][0]]
    longitude_delta = angle_difference(model_longitudes[points[1][1]],model_longitudes[
        points[0][1]])

    #clip latitude
    if ( latitude > 90):
        latitude = 90
    if (latitude < -90):
        latitude = -90

    for point in points:
        longitude_diff = (model_longitudes[point[1]] - longitude + 180) % 360 - 180
        factor = (
            1 - (abs(model_latitudes[point[0]] - latitude)) / latitude_delta
        ) * (1 - abs(longitude_diff) / longitude_delta)

        factors.append({"weight": factor, "lat_lon_indices": point})
    return factors


def enclosing_points(latitude, longitude,
                     model_latitudes,
                     model_longitudes):

    # use proper search algo for sorted arrays
    ii = numpy.searchsorted(  model_latitudes, latitude,side='right' )
    if ii==0:
        ilat = (0,1)
    elif ii == len(model_latitudes):
        ilat = (ii - 2, ii-1)
    else:
        ilat = (ii-1,ii)

    if model_longitudes[0] < 0:
        # longitudes are assumed in [-180,180)
        longitude = (longitude + 180)% 360 - 180
    else:
        # longitudes are assumed in [0, 360)
        longitude = longitude % 360

    if longitude > model_longitudes[-1]:
        longitude = longitude-360

    ii = numpy.searchsorted( model_longitudes, longitude, side='right' )
    if ii == 0:
        ilon = (len(model_longitudes)-1,ii)
    elif ii == len(model_longitudes):
        ilon = (len(model_longitudes)-1,ii)
    else:
        ilon = (ii - 1,ii)

    points = []
    for ii in ilat:
        for jj in ilon:
            points.append((ii, jj))

    return points

def interp_latitude_longitude(obs_time: list[datetime], lat_or_lon,
                              time: List[datetime]):
    # Interpolate latitude or longitude in time making sure we interpolate
    # correctly across the dateline

    # convert times to epochs for interpolation
    obs_time = [x.timestamp() for x in obs_time]
    time = [x.timestamp() for x in time]

    # Make sure we are in [0,360]
    lat_or_lon = numpy.array(lat_or_lon % 360)

    # unwrap to continuous vector
    lat_or_lon = numpy.array(
        numpy.unwrap(lat_or_lon, discont=360, period=360))

    # interpolate and return to [-180,180) range
    return (interp(time, obs_time, lat_or_lon, right=numpy.nan,
                  left=numpy.nan) + 180) % 360 - 180


def angle_difference( angle_to_subtract_from, angle_to_subtract, period=360 ):
    return (angle_to_subtract_from-angle_to_subtract + period/2)%period - period/2