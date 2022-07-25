import logging
import sys
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())

# External API tools

from .modeldata import sofarspectralapi
from roguewave.modeldata.sofarspectralapi import load_sofar_spectral_file

# Wave spectrum

## object definition
from .wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .wavespectra.spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .wavespectra import spectrum1D, spectrum2D

## estimators
from .wavespectra.estimators import convert_to_1d_spectrum,convert_to_2d_spectrum

## save/load
from .wavespectra.io import load_spectrum, save_spectrum # Depricated
from .io.io import save,load

## Partitioning
from .wavespectra.partitioning.wavefields import (
    get_bulk_partitions_from_spectral_partitions,
    get_spectral_partitions_from_2dspectra)
from .wavespectra.partitioning.observations import \
    get_bulk_partitions_from_observations, \
    get_spectral_partitions_from_observations

def set_log_to_console(level=logging.INFO):
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.setLevel(level)

def set_log_to_file(filename, level=logging.INFO):
    logger.addHandler(logging.FileHandler(filename))
    logger.setLevel(level)