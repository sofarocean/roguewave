import logging
import sys

# Logging setup
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())

# Caching
import roguewave.filecache as filecache

# External API tools
from .modeldata import sofarspectralapi
from roguewave.modeldata.sofarspectralapi import load_sofar_spectral_file

# model time
from roguewave.modeldata.timebase import TimeSliceLead, \
    timebase_forecast,TimeSliceForecast, TimeSliceEvaluation, TimeSliceAnalysis

# Interpolate
from .interpolate.dataset import interpolate_dataset, tracks_as_dataset
from .interpolate.geometry import TrackSet,Track,Cluster

# Wave spectrum
from .wavespectra.spectrum1D import WaveSpectrum1D
from .wavespectra.spectrum2D import WaveSpectrum2D
from .wavespectra import spectrum1D, spectrum2D

from .wavespectra.estimators import convert_to_1d_spectrum, \
    convert_to_2d_spectrum

# save/load
from .io.io import save, load

# Restart Files:
from roguewave.wavewatch3 import RestartFile
from roguewave.wavewatch3 import open_restart_file, \
    write_partial_restart_file, write_restart_file, \
    reassemble_restart_file_from_parts

# Partitioning
from .wavespectra.partitioning.wavefields import (
    get_bulk_partitions_from_spectral_partitions,
    get_spectral_partitions_from_2dspectra)
from .wavespectra.partitioning.observations import \
    get_bulk_partitions_from_observations, \
    get_spectral_partitions_from_observations

# Model data
from .modeldata.open_remote import open_remote_dataset
from .modeldata.modelinformation import model_timebase, model_valid_time, \
    available_models, list_available_variables
from roguewave.modeldata.extract import extract_from_remote_dataset
from roguewave.colocate.bulk import colocate_model_spotter, \
    colocated_tracks_as_dataset
from roguewave.colocate.spectra import colocate_model_spotter_spectra
from roguewave.modeldata.keygeneration import generate_uris



def set_log_to_file(filename, level=logging.INFO):
    logger.addHandler(logging.FileHandler(filename))
    logger.setLevel(level)


def set_level(level):
    if isinstance(level, int):
        logger.setLevel(level)

    elif isinstance(level, str):
        if level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif level == 'info':
            logger.setLevel(logging.INFO)
        elif level == 'warning':
            logger.setLevel(logging.WARNING)
        else:
            raise ValueError(f'unknown logging level {level}')
    else:
        raise ValueError(f'unknown logging level {level}')


def set_log_to_console(level=logging.INFO):
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    set_level(level)
