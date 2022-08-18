from typing import Mapping, Sequence, Union
from pandas import DataFrame
from roguewave.spotterapi import get_spectrum
from roguewave.interpolate.dataset import tracks_as_dataset
from roguewave.modeldata.open_remote_restart_files import open_remote_restart_file
from roguewave.interpolate.dataframe import interpolate_dataframe_time
from roguewave.modeldata.timebase import TimeSlice
from roguewave.interpolate.geometry import TrackSet
from roguewave.wavespectra import wave_spectra_as_data_array


# =============================================================================
def colocate_model_spotter_spectra(
        variable: str,
        spotter_ids: Sequence[str],
        time_slice:TimeSlice,
        model_name: str,
        model_definition: str,
        cache: bool = True,
        cache_name: str = None,
        timebase:str = 'native',
) -> Mapping[str,Mapping[str,DataFrame]]:

    # dataset = open_remote_dataset(
    #     variable=variable,
    #     time_slice=time_slice,
    #     model_name=model_name,
    #     cache_name=cache_name
    # )
    stack = open_remote_restart_file(variable,time_slice, model_name,
                            model_definition,cache_name=cache_name,cache=cache)
    spotters = get_spectrum(spotter_ids, time_slice.start_time,
                                  time_slice.end_time,cache=cache)

    if timebase == 'native':
        trackset = TrackSet.from_spotters(spotters).interpolate(stack.time)
    elif timebase == 'spotter':
        trackset = TrackSet.from_spotters(spotters)
    else:
        trackset = TrackSet.from_spotters(spotters).interpolate(stack.time)

    for spotter_id in spotters:
        spotters[spotter_id] = wave_spectra_as_data_array(spotters[spotter_id])

    model = stack.interpolate_tracks(trackset)
    out = {}
    for spotter in spotters:
        out[spotter] = {'spotter': spotters[spotter],
                        'model': model[spotter]}

    return out

