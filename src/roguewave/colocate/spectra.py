from typing import Dict, Sequence, Tuple, Union
from roguewave.spotterapi import get_spectrum
from roguewave.modeldata.open_remote_restart_files import \
    open_remote_restart_file
from roguewave.modeldata.timebase import TimeSlice
from roguewave.interpolate.geometry import TrackSet
from roguewave.tools.time import to_datetime64
from roguewave import FrequencyDirectionSpectrum
from xarray import Dataset, concat


# =============================================================================
def colocate_model_spotter_spectra(
        variable: str,
        spotter_ids: Sequence[str],
        time_slice: TimeSlice,
        model_name: str,
        model_definition: str,
        cache: bool = True,
        cache_name: str = None,
        spectral_domain: str = 'model',
        timebase: str = 'model',
        return_as_dataset = True
) -> Union[ Tuple[FrequencyDirectionSpectrum,FrequencyDirectionSpectrum]
           ,Tuple[Dict[str,FrequencyDirectionSpectrum],
                  Dict[str,FrequencyDirectionSpectrum]]]:

    if return_as_dataset:
        if (not timebase == 'model') and (len(spotter_ids) > 1):
            raise ValueError('Cannot return as an xarray dataset if the '
                             'time base is native or spotter')

    stack = open_remote_restart_file(variable, time_slice, model_name,
                                     model_definition, cache_name=cache_name,
                                     cache=cache)
    spotters = get_spectrum(spotter_ids, time_slice.start_time,
                            time_slice.end_time, cache=cache)

    if timebase == 'spotter':
        trackset = TrackSet.from_spotters(spotters)
    else:
        trackset = TrackSet.from_spotters(spotters).interpolate(stack.time)

    for spotter_id in spotters:
        spotter_data = spotters[spotter_id].as_frequency_direction_spectrum(
            number_of_directions=stack.number_of_directions
        )

        if timebase == 'model':
            spotter_data = spotter_data.interpolate(
                {'time':to_datetime64(trackset.tracks[spotter_id].time)})

        if spectral_domain == 'model':
            spotter_data =spotter_data.interpolate(
                coordinates={
                    'frequency':stack.frequency,
                     'direction': stack.direction
                }
            )
        spotters[spotter_id] = spotter_data

    print('Get model data along spotter tracks:')
    model = stack.interpolate_tracks(trackset)
    if spectral_domain == 'spotter':
        for spotter_id in model:
            model[spotter_id] = model[spotter_id].interpolate(coordinates={
                    'frequency': spotters['spotter_id']['frequency'],
                     'direction': spotters['spotter_id']['direction']
                })

    if return_as_dataset:
        m = concat( [x.dataset for x in model.values()],'spotter_id')
        model = m.assign_coords(spotter_id=spotter_ids)
        model = FrequencyDirectionSpectrum(model)

        s = concat( [x.dataset for x in spotters.values()],'spotter_id')
        spotters = s.assign_coords(spotter_id=spotter_ids)
        spotters = FrequencyDirectionSpectrum(spotters)

    return model, spotters
