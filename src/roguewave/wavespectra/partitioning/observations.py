from .wavefields import wave_fields_from, filter_fields, create_graph, \
    bulk_parameters_partitions
from .partitioning import partition_spectrum, default_partition_config
from roguewave.wavespectra import spec1d_from_spec2d, WaveSpectrum1D
from roguewave.wavespectra.operators import spectrum1D_time_filter
from roguewave.wavespectra.estimators import spect2d_from_spec1d
from typing import List
from datetime import timedelta
from pandas import DataFrame
from roguewave.wavespectra import WaveSpectrum2D

default_config = {'smoothInTime': False,
                  'estimator': {
                      'method': 'mem2',
                      'numberOfDirections': 36,
                      'frequencySmoothing': False,
                      'smoothingLengthscale': 1
                  },
                  'partitionConfig': default_partition_config,
                  'fieldFiltersettings': {
                      'filter': True,
                      'maxDeltaPeriod': 2,
                      'maxDeltaDirection': 20
                  }
                  }

def partition_observations_bulk(spectra: List[WaveSpectrum1D],
                           minimum_duration: timedelta,
                           config=None)->List[DataFrame]:
    wave_fields = partition_observations_spectra(spectra, minimum_duration,config = config)
    return bulk_parameters_partitions(wave_fields)

def partition_observations_spectra(spectra: List[WaveSpectrum1D],
                           minimum_duration: timedelta,
                           config=None)->List[List[WaveSpectrum2D]]:
    if config:
        for key in config:
            assert key in default_config, f"{key} is not a valid conficuration entry"

        config = default_config | config
    else:
        config = default_config

    # Step 1: Pre-Processing:

    # Prior to constructing spectra - smoothing the wave field can help create
    # more stable results.
    if config['smoothInTime']:
        spectra = spectrum1D_time_filter(spectra)

    # Step 2: Create 2D wavefields from 1D spectra using a spectral estimator
    spectra2D = [spect2d_from_spec1d(
        spectrum,
        method=config['estimator']['method'],
        number_of_directions=config['estimator']['numberOfDirections'],
        frequency_smoothing=config['estimator']['frequencySmoothing'],
        smoothing_lengthscale=config['estimator']['smoothingLengthscale']
    ) for spectrum in spectra]

    # Step 3: Partition the data
    raw_partitions = []
    for spectrum in spectra2D:
        partitions, _ = partition_spectrum(spectrum, config['partitionConfig'])
        raw_partitions.append(partitions)

    # Step 4: create a graph
    graph = create_graph(raw_partitions, minimum_duration)

    # Step 5: create wave field from the graph
    wave_fields = wave_fields_from(graph)

    # Step 6: Postprocessing

    # Apply a filter on the bulk parameters
    if config['fieldFiltersettings']['filter']:
        wave_fields = filter_fields(
            wave_fields,
            min_duration=minimum_duration,
            max_delta_period=config['fieldFiltersettings']['maxDeltaPeriod'],
            max_delta_direction=config['fieldFiltersettings'][
                'maxDeltaDirection']
        )

    return wave_fields
