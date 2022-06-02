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
from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D
from .wavespectrum import WaveSpectrum
from . import spectrum1D, spectrum2D
from typing import Union, Dict, List
import gzip
import json

_UNION = Union[
    WaveSpectrum1D, WaveSpectrum2D, List[WaveSpectrum1D], List[WaveSpectrum2D],
    Dict[str, List[WaveSpectrum1D]], Dict[str, List[WaveSpectrum2D]],
    Dict[str, List[List[WaveSpectrum2D]]], List[List[WaveSpectrum2D]]]


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
    with gzip.open(filename, 'rb') as file:
        data = file.read().decode('utf-8')

    data = _deserialize(json.loads(data))
    return data


def _deserialize(data):
    """
    Convert the JSON representation of (nested) wavespectra back into native
    python representation.
    :param data: serialized data
    :return: native python representation.
    """

    if isinstance(data, (dict)):
        # if the dictionary contains "directions" or "frequency" as keys these
        # are serialized spectral objects/
        if 'directions' in data:
            # if directions -> create a wavespectrum2d. The keys directly map
            # to the constructor arguments.
            return spectrum2D(**data)
        elif 'frequency' in data:
            # if not directions but has frequency -> create a wavespectrum1d.
            # The keys directly map to the constructor arguments.
            return spectrum1D(**data)
        else:
            # Otherwise- this is a nested object where each key represents data
            # at a different Spotter/location. Loop over all keys and call the
            # function recursively.
            output = {}
            for key in data:
                output[key] = _deserialize(data[key])
            return output
    elif isinstance(data, list):
        # This is a nested object where each entry represents data
        # at a different Spotter/location. Loop over all entries and call the
        # function recursively.
        output = []
        for item in data:
            output.append(_deserialize(item))
        return output
    else:
        raise Exception('Cannot convert to json compatible type')


def _serialize(_input):
    """
    Convert nested objects into a form that can be serialized as JSON. The lowest
    level data structure is either  WaveSpectrum1D or WaveSpectrum2D.

    :param _input: see load_spectrum
    :return: object that can be json Serialized
    """

    if isinstance(_input, WaveSpectrum):
        # If the input is of type spectrum convert the spectra into a JSON form
        if isinstance(_input, WaveSpectrum1D):
            return _input._create_wave_spectrum_input()
        elif isinstance(_input, WaveSpectrum2D):
            return _input._create_wave_spectrum_input()
    elif isinstance(_input, (dict)):
        # If the input is of type dict, loop over all the keys, and call
        # function recursively on each of the elements. The output is stored
        # in a dictionary with the same keys- but now with content that can
        # be serialized.
        output = {}
        for key in _input:
            output[key] = _serialize(_input[key])
        return output
    elif isinstance(_input, list):
        # If the input is of type list, loop over all the entries, and call
        # function recursively on each of the elements. The output is stored
        # in a list of the same length and ordering- but now with content that can
        # be serialized.
        output = []
        for item in _input:
            output.append(_serialize(item))
        return output
    else:
        # Otherwise raise error
        raise Exception('Cannot convert to json compatible type')


def save_spectrum(_input: _UNION, filename: str):
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
    output = _serialize(_input)

    data = json.dumps(output)
    with gzip.open(filename, 'wb') as file:
        file.write(data.encode('utf-8'))
