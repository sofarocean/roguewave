from typing import Mapping, Sequence, Union
from pandas import DataFrame
from roguewave.modeldata.open_remote import open_remote_dataset
from roguewave.spotterapi.spotterapi import get_bulk_wave_data
from roguewave.interpolate.dataset import interpolate_dataset
from roguewave.modeldata.extract import extract_from_remote_dataset
from roguewave.interpolate.dataframe import interpolate_dataframe_time
from roguewave.modeldata.timebase import TimeSlice
from roguewave.interpolate.geometry import TrackSet

# =============================================================================
def colocate_model_spotter(
        variable: Union[Sequence[str], str],
        spotter_ids: Sequence[str],
        time_slice:TimeSlice,
        model_name: str,
        cache_name: str = None,
        parallel: bool=True,
        timebase:str = 'native',
        slice_remotely=False
) -> Mapping[str,Mapping[str,DataFrame]]:
    """

    :param variable: name of the variable of interest
    :param spotter_ids:
    :param init_time: init time of the forecast of interest
    :param duration: maximum lead time of interest
    :param model_name: model name
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :return:
    """

    # dataset = open_remote_dataset(
    #     variable=variable,
    #     time_slice=time_slice,
    #     model_name=model_name,
    #     cache_name=cache_name
    # )

    spotters = get_bulk_wave_data(spotter_ids, time_slice.start_time,
                                  time_slice.end_time,
                                  convert_to_sofar_model_names=True)

    data =  extract_from_remote_dataset( spotters, variable,
                            time_slice,model_name,slice_remotely=slice_remotely,
                                parallel=parallel, cache_name=cache_name  )
    out = {}
    for spotter_id in spotter_ids:
        s = spotters[spotter_id] # type: DataFrame
        m = data[spotter_id] # type: DataFrame
        if timebase.lower() == 'native':
            pass
        elif timebase.lower() == 'observed' or timebase.lower() == 'spotter':
            m = interpolate_dataframe_time(m, s.index.values)
        elif timebase.lower() == 'model':
            s = interpolate_dataframe_time(s, m.index.values)
        else:
            raise ValueError(f'Unknown timebase {timebase}, must be one of: '
                             f'native, observed, model, or spotter')

        out[spotter_id] = {
            'model':m,
            'spotter':s
        }
    return out
