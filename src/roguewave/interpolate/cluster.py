from roguewave.interpolate.points import interpolate_points_nd
from roguewave.interpolate.geometry import Cluster
import numpy

def interpolate_cluster(
    latitude: numpy.ndarray,
    longitude: numpy.ndarray,
    cluster:Cluster,
    get_data,
    period_data=None,
    discont=None
    ):

    coordinates = [('latitude', latitude), ('longitude', longitude)]
    _points = {'latitude': cluster.latitude,
               'longitude': cluster.longitude}

    interpolated_points = interpolate_points_nd(
        coordinates=coordinates,
        points=_points,
        periodic_coordinates={'longitude':360},
        get_data=get_data,
        period_data=period_data,
        discont=discont
    )

    output = {}
    for index,name in enumerate(cluster.points.keys()):
        output[name] = interpolated_points[index]

    return output