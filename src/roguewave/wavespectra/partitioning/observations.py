from .wavefields import wave_fields_from, filter_fields, create_graph, \
    bulk_parameters_partitions
from .partitioning import partition_spectrum, default_partition_config
from roguewave.wavespectra import spec1d_from_spec2d, WaveSpectrum1D
from roguewave.wavespectra.operators import spectrum1D_time_filter
from roguewave.wavespectra.estimators import spec2d_from_spec1d
from typing import List, Dict, Union
from datetime import timedelta
from pandas import DataFrame
from roguewave.wavespectra import WaveSpectrum2D
from typing import overload

default_config = {
    'smoothInTime': False,
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


# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
@overload
def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: Dict[str, List[List[WaveSpectrum2D]]]
) -> Dict[str, List[DataFrame]]: ...

@overload
def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: List[List[WaveSpectrum2D]]
) -> List[DataFrame]: ...

@overload
def get_spectral_partitions_from_observations(
        spectra: Dict[str, List[WaveSpectrum1D]],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Dict[str, List[List[WaveSpectrum2D]]]: ...


@overload
def get_spectral_partitions_from_observations(
        spectra: List[WaveSpectrum1D],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> List[List[WaveSpectrum2D]]: ...


@overload
def get_bulk_partitions_from_observations(
        spectra: Dict[str, List[WaveSpectrum1D]],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Dict[str, List[DataFrame]]: ...


@overload
def get_bulk_partitions_from_observations(
        spectra: List[WaveSpectrum1D],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> List[DataFrame]: ...


# -----------------------------------------------------------------------------
#                       Implementationb
# -----------------------------------------------------------------------------
def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: Union[Dict[str, List[List[WaveSpectrum2D]]], List[
            List[WaveSpectrum2D]]]) -> Union[Dict[
                                                 str, List[DataFrame]], List[
                                                 DataFrame]]:
    if isinstance(spectral_partitions, dict):
        output = {}
        for key in spectral_partitions:
            output[key] = bulk_parameters_partitions(spectral_partitions[key])

    elif isinstance(spectral_partitions, list):
        output = bulk_parameters_partitions(spectral_partitions)
    else:
        raise Exception('Cannot process input')

    return output


def get_bulk_partitions_from_observations(
        spectra: Union[Dict[str, List[WaveSpectrum1D]], List[WaveSpectrum1D]],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Union[Dict[str, List[DataFrame]], List[DataFrame]]:
    #
    spectral_partitions = get_spectral_partitions_from_observations(
        spectra, minimum_duration, config, verbose)

    if isinstance(spectra, dict):
        output = {}
        for key in spectral_partitions:
            output[key] = bulk_parameters_partitions(spectral_partitions[key])

    elif isinstance(spectra, list):
        output = bulk_parameters_partitions(spectral_partitions)
    else:
        raise Exception('Cannot process input')

    return output


def get_spectral_partitions_from_observations(
        spectra: Union[Dict[str, List[WaveSpectrum1D]], List[WaveSpectrum1D]],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Union[
    Dict[str, List[List[WaveSpectrum2D]]], List[List[WaveSpectrum2D]]]:
    #

    if isinstance(spectra, dict):
        output = {}
        for key, item in spectra.items():
            output[key] = partition_observations_spectra(item,
                                                         minimum_duration,
                                                         config, verbose)
    elif isinstance(spectra, list):
        output = partition_observations_spectra(spectra,
                                                minimum_duration,
                                                config, verbose)
    else:
        raise Exception('Cannot process input')

    return output


def _print(verbose, *narg, **kwargs):
    if verbose:
        print(*narg, **kwargs)


def partition_observations_bulk(spectra: List[WaveSpectrum1D],
                                minimum_duration: timedelta,
                                config=None, verbose=False) -> List[DataFrame]:
    wave_fields = partition_observations_spectra(spectra, minimum_duration,
                                                 config=config,
                                                 verbose=verbose)
    return bulk_parameters_partitions(wave_fields)


def partition_observations_spectra(spectra: List[WaveSpectrum1D],
                                   minimum_duration: timedelta,
                                   config=None, verbose=False) -> List[
    List[WaveSpectrum2D]]:
    if config:
        for key in config:
            assert key in default_config, f"{key} is not a valid conficuration entry"

        config = default_config | config
    else:
        config = default_config

    # Step 1: Pre-Processing
    # :
    # Prior to constructing spectra - smoothing the wave field can help create
    # more stable results.

    _print(verbose, '*** Partitioning Data ***\n' + 80 * '-' + '\n')
    if config['smoothInTime']:
        _print(verbose, ' - Smoothing in time')
        spectra = spectrum1D_time_filter(spectra)

    # Step 2: Create 2D wavefields from 1D spectra using a spectral estimator
    spectra2D = []
    _print(verbose, ' - Create 2D Spectra')
    for index, spectrum in enumerate(spectra):
        _print(verbose, f'\t {index:05d} out of {len(spectra)}')
        spectra2D.append(
            spec2d_from_spec1d(
                spectrum,
                method=config['estimator']['method'],
                number_of_directions=config['estimator']['numberOfDirections'],
                frequency_smoothing=config['estimator']['frequencySmoothing'],
                smoothing_lengthscale=config['estimator'][
                    'smoothingLengthscale']
            )
        )

    # Step 3: Partition the data
    _print(verbose, ' - Partition Data')
    raw_partitions = []
    for index, spectrum in enumerate(spectra2D):
        _print(verbose, f'\t {index:05d} out of {len(spectra2D)}')
        partitions, _ = partition_spectrum(spectrum, config['partitionConfig'])
        raw_partitions.append(partitions)

    # Step 4: create a graph
    _print(verbose, ' - Create Graph')
    graph = create_graph(raw_partitions, minimum_duration)

    # Step 5: create wave field from the graph
    _print(verbose, ' - Create Wave Fields From Graph')
    wave_fields = wave_fields_from(graph)

    # Step 6: Postprocessing

    # Apply a filter on the bulk parameters
    _print(verbose, ' - Apply Bulk Filter')
    if config['fieldFiltersettings']['filter']:
        wave_fields = filter_fields(
            wave_fields,
            min_duration=minimum_duration,
            max_delta_period=config['fieldFiltersettings']['maxDeltaPeriod'],
            max_delta_direction=config['fieldFiltersettings'][
                'maxDeltaDirection']
        )
    _print(verbose, '*** Done ***\n' + 80 * '-' + '\n\n')
    return wave_fields
