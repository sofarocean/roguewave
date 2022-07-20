import numpy

def interpolation_weights(latitude,
                          longitude,
                          model_latitudes,
                          model_longitudes):

    model_longitudeStepSize = numpy.abs(numpy.diff(model_longitudes)[0])
    model_latitudeStepSize = numpy.abs(numpy.diff(model_latitudes)[0])
    points = enclosing_points(latitude, longitude, model_latitudes, model_longitudes)

    factors = []
    for point in points:

        longitude_diff = (model_longitudes[point[1]] - longitude + 180) % 360 - 180
        factor = (
            1 - (abs(model_latitudes[point[0]] - latitude)) / model_latitudeStepSize
        ) * (1 - abs(longitude_diff) / model_longitudeStepSize)

        factors.append({"weight": factor, "lat_lon_indices": point})
    return factors


def enclosing_points(latitude, longitude,
                     model_latitudes,
                     model_longitudes):

    # use proper search algo for sorted arrays
    ii = numpy.searchsorted(  model_latitudes, latitude,side='right' )

    if ii==0:
        ilat = (
            0,
            1
        )
    else:
        ilat = (
            ii-1,
            ii
        )

    longitude = (longitude + 180)% 360 - 180
    if longitude > model_longitudes[-1]:
        longitude = longitude-360

    ii = numpy.searchsorted( model_longitudes, longitude, side='right' )
    if ii == 0:
        ilon = (
            len(model_longitudes)-1,
            ii
        )
    else:
        ilon = (
            ii - 1,
            ii
        )

    points = []
    for ii in ilat:
        for jj in ilon:
            points.append((ii, jj))

    # Don't count the same point twice, which can happen if a point is on the edge of a grid
    points = list(set(points))
    return points