from .externaldata import sofarspectralapi
from .externaldata.spotter import get_spectrum_from_sofar_spotter_api
from .wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .wavespectra.spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .wavespectra import spectrum1D, spectrum2D
from .wavespectra.estimators import convert_to_1d_spectrum,convert_to_2d_spectrum
from .wavespectra.io import load_spectrum, save_spectrum
from .wavespectra.partitioning.observations import \
    partition_observations_spectra, partition_observations_bulk
from .wavespectra.io import load_spectrum, save_spectrum
from .wavespectra.partitioning.observations import \
    get_bulk_partitions_from_observations, \
    get_spectral_partitions_from_observations, \
    get_bulk_partitions_from_spectral_partitions
