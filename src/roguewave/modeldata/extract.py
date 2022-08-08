import numpy
from pandas import DataFrame
from roguewave.tools.time import to_datetime64, to_datetime_utc
from roguewave.interpolate.cluster import interpolate_cluster
from xarray import DataArray, open_dataset
from roguewave.interpolate.geometry import ClusterStack, Cluster, Geometry, \
    convert_to_track_set, convert_to_cluster_stack
from .timebase import TimeSlice
from .modelinformation import _get_resource_specification
from .keygeneration import generate_uris
from roguewave.modeldata.open_remote import open_remote_dataset
from typing import Dict,List, Union
from roguewave.interpolate.dataset import interpolate_dataset
from roguewave.modeldata.remote_point_data import extract_clusters
BLOCKSIZE = 1024


def extract_from_remote_dataset(
        geometry:Geometry,
        variable,
        time_slice:TimeSlice,
        model_name:str,
        slice_remotely=False,
        parallel=True,
        cache_name=None
):

    if slice_remotely:
        return extract_clusters(geometry, variable, time_slice,model_name,
                                 parallell=parallel)

    else:
        dataset = open_remote_dataset( variable, time_slice, model_name,
                                       cache_name=cache_name)
        return interpolate_dataset(dataset,geometry,)


