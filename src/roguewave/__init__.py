# External API tools

from .externaldata import sofarspectralapi
from .externaldata.spotter import get_spectrum_from_sofar_spotter_api
from .externaldata.sofarspectralapi import load_sofar_spectral_file

# Wave spectrum

## object definition
from .wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .wavespectra.spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .wavespectra import spectrum1D, spectrum2D

## estimators
from .wavespectra.estimators import convert_to_1d_spectrum,convert_to_2d_spectrum

## save/load
from .wavespectra.io import load_spectrum, save_spectrum

## Partitioning
from .wavespectra.partitioning.wavefields import (
    get_bulk_partitions_from_spectral_partitions,
    get_spectral_partitions_from_2dspectra)
from .wavespectra.partitioning.observations import \
    get_bulk_partitions_from_observations, \
    get_spectral_partitions_from_observations