from pandas import DataFrame
from roguewave.interpolate.geometry import Geometry
from .timebase import TimeSlice
from roguewave.modeldata.open_remote import open_remote_dataset
from typing import Dict,List, Union
from roguewave.interpolate.dataset import interpolate_dataset
from roguewave.modeldata.remote_point_data import extract_clusters


def extract_from_remote_dataset(
        geometry:Geometry,
        variable:Union[List[str],str],
        time_slice:TimeSlice,
        model_name:str,
        slice_remotely=False,
        parallel=True,
        cache_name=None
) -> Dict[str, DataFrame]:
    """
    Extract timeseries at specified locations from the dataset identified by
    the model name and the timeslice.

    :param geometry: space(time) geometry we want to download from (e.g. a set
        of points, or a spotter track). For most use cases can be specified as

            geometry  =   ( [latitudes ] , [longitudes] )

        more advanced slicing will require the use of interpolation geometries
        defined in roguewave.interpolation.geometry.

    :param variable: name of the variable of interest. Can be a list in which
            case all variables in the list are retrieved.
    :param time_slice: time slice of interest.
    :param model_name: model name
    :param slice_remotely: (default False) if True Skip local cache and try to
    read directly from
        remote. This tries to avoid downloading full files and instead tries
        to only grab the data it needs. Typically slower.
        Does not work for grib files.
    :param parallel: If slice_remotely=True controls whether we download in
        parallel. To note- if parallel is enabled we need to make sure the
        main script is guarded by if __name__ == '__main__'.
    :param cache_name: name of local cache. If None, default cache setup will
        be used.

    :return: Dictionary where each geometry element is a key, and the value is
        a pandas dataframe with as index the time, and as columns the
        requested variables along the given geometry.
    """

    if slice_remotely:
        return extract_clusters(geometry, variable, time_slice,model_name,
                                 parallell=parallel)

    else:
        dataset = open_remote_dataset( variable, time_slice, model_name,
                                       cache_name=cache_name)
        return interpolate_dataset(dataset,geometry)


