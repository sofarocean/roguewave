"""
Contents: IO

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines that can be used to save and load Spectral data.

Functions:

- `load_spectrum`, load spectral data
- `save_spectrum`, save spectral data
- `_deserialize`, deserialize data
- `_serialize`, serialize data


How To Use This Module
======================
(See the individual functions for details.)

1. No need for direct import, use roguewave ``import roguewave`` or ``from roguewave import ...``.
2. Save data: save_spectrum(data, filename)
3. load data: data = load_spectrum(filename)

"""
from roguewave.io.io import save,load, _UNION

def load_spectrum(filename: str) -> _UNION:
    """
    Load spectral data as saved by "save_spectrum" from the given file and
    return a (nested) object. The precise format of the output depends on what
    was saved.

    :param filename: path to file to load.
    :return:
        - WaveSpectrum1D
          A single 1D spectrum

        - WaveSpectrum2D
          A single 2D spectrum

        - List[WaveSpectrum1D]
          List of 1D spectra

        - List[WaveSpectrum2D],
          List of 2D spectra.

        - Dict[str, List[WaveSpectrum1D]]
          Dictionary with Spotter/Observation name as keys, each containing a
          List of observed 1D spectra

        - Dict[str, List[WaveSpectrum2D]]
          Dictionary with Spotter/Observation name as keys, each containing a
          List of observed and reconstructed/computed 2D spectra

        - List[List[WaveSpectrum2D]]]
          List of partitioned data - each partition in itself is a list of
          2D wavespectra ordered in time.

        - Dict[str, List[List[WaveSpectrum2D]]]
          Dictionary with Spotter/Observation name as keys, each containing a list
          of partitioned data - each partition in itself is a list of
          2D wavespectra ordered in time.
    """

    return load(filename)


def save_spectrum(_input: _UNION, filename: str, format='json', separate_spotters_into_files=False):
    """
    Save spectral data, possible in nested form as returned by the spectral
    partition/reconstruction/etc. functions. Data is saved in JSON form
    compressed with gzip.

    :param _input:
        - WaveSpectrum1D
          A single 1D spectrum

        - WaveSpectrum2D
          A single 2D spectrum

        - List[WaveSpectrum1D]
          List of 1D spectra

        - List[WaveSpectrum2D],
          List of 2D spectra.

        - Dict[str, List[WaveSpectrum1D]]
          Dictionary with Spotter/Observation name as keys, each containing a
          List of observed 1D spectra

        - Dict[str, List[WaveSpectrum2D]]
          Dictionary with Spotter/Observation name as keys, each containing a
          List of observed and reconstructed/computed 2D spectra

        - List[List[WaveSpectrum2D]]]
          List of partitioned data - each partition in itself is a list of
          2D wavespectra ordered in time.

        - Dict[str, List[List[WaveSpectrum2D]]]
          Dictionary with Spotter/Observation name as keys, each containing a list
          of partitioned data - each partition in itself is a list of
          2D wavespectra ordered in time.

    :param filename: path to save data.
    :return: None
    """

    return save(_input,filename,format,separate_spotters_into_files)