from typing import Mapping, Sequence, Union
from xarray import Dataset
from roguewave.spotterapi import get_spectrum
from roguewave.modeldata.open_remote_restart_files import \
    open_remote_restart_file
from roguewave.interpolate.dataset import \
    interpolate_dataset_along_axis, \
    interpolate_dataset_grid
from roguewave.modeldata.timebase import TimeSlice
from roguewave.interpolate.geometry import TrackSet
from roguewave.wavespectra import wave_spectra_as_data_set
from roguewave.tools.time import to_datetime64


# =============================================================================
def colocate_model_spotter_spectra(
        variable: str,
        spotter_ids: Sequence[str],
        time_slice: TimeSlice,
        model_name: str,
        model_definition: str,
        cache: bool = True,
        cache_name: str = None,
        spectral_domain: str = 'native',
        timebase: str = 'native',
) -> Mapping[str, Mapping[str, Dataset]]:

    stack = open_remote_restart_file(variable, time_slice, model_name,
                                     model_definition, cache_name=cache_name,
                                     cache=cache)
    spotters = get_spectrum(spotter_ids, time_slice.start_time,
                            time_slice.end_time, cache=cache)

    if timebase == 'native':
        trackset = TrackSet.from_spotters(spotters).interpolate(stack.time)
    elif timebase == 'spotter':
        trackset = TrackSet.from_spotters(spotters)
    else:
        trackset = TrackSet.from_spotters(spotters).interpolate(stack.time)

    for spotter_id in spotters:
        spotter_data = wave_spectra_as_data_set(spotters[spotter_id])

        if timebase == 'model':
            spotter_data = interpolate_dataset_along_axis(
                to_datetime64(stack.time),spotter_data,'time')

        if spectral_domain == 'model':
            spotter_data =interpolate_dataset_grid(
                coordinates={
                    'frequency':stack.frequency,
                     'direction': stack.direction
                },
                data_set=spotter_data
            )

            
        spotters[spotter_id] = spotter_data

    model = stack.interpolate_tracks(trackset)
    out = {}
    for spotter in spotters:
        out[spotter] = {'spotter': spotters[spotter],
                        'model': model[spotter]}

    return out
