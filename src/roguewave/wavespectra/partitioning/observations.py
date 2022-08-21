from .wavefields import bulk_parameters_partitions, DEFAULT_CONFIG_PARTITION_SPECTRA
from .wavefields import partition_spectra
from typing import List, Dict, Union
from datetime import timedelta
from pandas import DataFrame
from roguewave.wavespectra import FrequencySpectrum, FrequencyDirectionSpectrum
from typing import overload
from roguewave import logger

default_config = {
    'smoothInTime': False,
    'estimator': {
        'method': 'mem2',
        'numberOfDirections': 36,
        'frequencySmoothing': False,
        'smoothingLengthscale': 1
    },
    'partition_spectra':DEFAULT_CONFIG_PARTITION_SPECTRA
}


# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
@overload
def get_spectral_partitions_from_observations(
        spectra: Dict[str, FrequencySpectrum],
        minimum_duration: timedelta,
        config=None) -> Dict[str, List[FrequencyDirectionSpectrum]]: ...


@overload
def get_spectral_partitions_from_observations(
        spectra: FrequencySpectrum,
        minimum_duration: timedelta,
        config=None) -> List[FrequencyDirectionSpectrum]: ...


@overload
def get_bulk_partitions_from_observations(
        spectra: Dict[str, FrequencySpectrum],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Dict[str, List[DataFrame]]: ...


@overload
def get_bulk_partitions_from_observations(
        spectra: FrequencySpectrum,
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> List[DataFrame]: ...


# -----------------------------------------------------------------------------
#                       Implementation
# -----------------------------------------------------------------------------


def get_bulk_partitions_from_observations(
        spectra: Union[Dict[str, FrequencySpectrum], FrequencySpectrum],
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
        spectra: Union[Dict[str, FrequencySpectrum], FrequencySpectrum],
        minimum_duration: timedelta,
        config=None) -> Union[
    Dict[str, List[FrequencyDirectionSpectrum]],
        List[FrequencyDirectionSpectrum]]:
    #

    if isinstance(spectra, dict):
        output = {}
        number = len(spectra)
        index = 0
        for key, item in spectra.items():
            index += 1
            if item is None:
                continue
            logger.info( f'{index:05d} out of {number:05d} spotter: {key}' )
            output[key] = partition_observations_spectra(item,
                                                         minimum_duration,
                                                         config,indent='    ')
    elif isinstance(spectra, list):
        output = partition_observations_spectra(spectra,
                                                minimum_duration,
                                                config)
    else:
        raise Exception('Cannot process input')

    return output


def partition_observations_spectra(spectra: FrequencySpectrum,
                                   minimum_duration: timedelta,
                                   config=None, indent='') -> List[
    FrequencyDirectionSpectrum]:
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

    if config['smoothInTime']:
        pass

    # Step 2: Create 2D wavefields from 1D spectra using a spectral estimator

    logger.info( indent + ' - Create 2D Spectra')
    spectra2D = spectra.as_frequency_direction_spectrum(
        number_of_directions=36)
    return partition_spectra(
        spectra2D, minimum_duration, config=config['partition_spectra'],indent=indent)


