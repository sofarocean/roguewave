from roguewave.interpolate.points import interpolate_points_nd
from roguewave.interpolate.nd_interp import NdInterpolator
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

    interpolator = NdInterpolator(
        get_data=get_data,
        data_coordinates=coordinates,
        data_shape=( len(latitude),len(longitude) ),
        interp_coord_names=list(_points.keys()),
        interp_index_coord_name='latitude',
        data_periodic_coordinates={'longitude':360},
        data_period=period_data,
        data_discont=discont
    )

    interpolated_points = interpolator.interpolate(_points)

    output = {}
    for index,name in enumerate(cluster.points.keys()):
        output[name] = interpolated_points[index]

    return output