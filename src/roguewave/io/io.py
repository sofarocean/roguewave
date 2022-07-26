"""
Contents: IO

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines that can be used to save and load data.

Functions:



How To Use This Module
======================
(See the individual functions for details.)



"""
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D
from roguewave.wavespectra.spectrum2D import WaveSpectrum2D
from roguewave.metoceandata import WaveBulkData, WindData, SSTData,BarometricPressure
from pandas import DataFrame, read_json
from typing import Union, Dict, List
from datetime import datetime
import gzip
import json
import os
import numpy
import base64


_UNION = Union[
    WaveSpectrum1D,
    WaveSpectrum2D,
    List[WaveSpectrum1D],
    List[WaveSpectrum2D],
    Dict[str, List[WaveSpectrum1D]],
    Dict[str, List[WaveSpectrum2D]],
    Dict[str, List[List[WaveSpectrum2D]]],
    List[List[WaveSpectrum2D]],
    List[List[DataFrame]],
    Dict[int,List[DataFrame]],
    Dict[str,List[WaveBulkData]],
    Dict[str,DataFrame]
]


def _b64_encode_numpy(val:numpy.ndarray)->dict:
    if val.dtype == numpy.float64:
        val=val.astype(dtype=numpy.float32)

    if val.dtype == numpy.int64:
        val=val.astype(dtype=numpy.int32)

    dtype = val.dtype.str
    shape = val.shape
    data = base64.b64encode(val)

    return {"__class__":"ndarray", "data":{"dtype":dtype,"shape":shape,"data":data.decode('utf8')}}

def _b64_decode_numpy(data):
    decoded_bytes = base64.b64decode(bytes(data['data'],encoding='utf-8'))
    dtype = numpy.dtype(data['dtype'])
    array = numpy.frombuffer(decoded_bytes,dtype=dtype)
    return numpy.reshape(array,data['shape'])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return _b64_encode_numpy(obj)
        elif isinstance(obj, numpy.int32):
            return int(obj)
        elif isinstance(obj, numpy.float32):
            return float(obj)
        elif isinstance(obj, numpy.int64):
            return int(obj)
        elif isinstance(obj, numpy.float64):
            return float(obj)
        elif isinstance(obj, datetime):
            return {'__class__':'datetime', 'data':datetime.isoformat(obj)}
        elif isinstance(obj,DataFrame):
            return {'__class__':'DataFrame', 'data': obj.to_json(date_unit='s')}
        elif isinstance(obj, WaveSpectrum1D):
            return {'__class__': 'WaveSpectrum1D', 'data':obj._create_wave_spectrum_input()}
        elif isinstance(obj, WaveSpectrum2D):
            return {'__class__': 'WaveSpectrum2D', 'data':obj._create_wave_spectrum_input()}
        elif isinstance(obj, WaveBulkData):
            return {'__class__': 'WaveBulkData', 'data':obj.as_dict()}
        elif isinstance(obj, WindData):
            return {'__class__': 'WindData', 'data':obj.as_dict()}
        elif isinstance(obj, SSTData):
            return {'__class__': 'SSTData', 'data':obj.as_dict()}
        elif isinstance(obj, BarometricPressure):
            return {'__class__': 'BarometricPressure', 'data':obj.as_dict()}
        else:
            return json.JSONEncoder.default(self, obj)

def object_hook(dictionary:dict):
    if '__class__' in dictionary:
        if dictionary['__class__'] == 'DataFrame':
            #
            #print(dictionary['data'])
            df = read_json(dictionary['data'])
            if 'timestamp' in df:
                df['timestamp'] = df['timestamp'].apply(
                    lambda x: x.tz_localize('utc'))
            else:
                df.index = [ x.tz_localize('utc') for x in df.index]
            return df
        elif dictionary['__class__'] == 'WaveSpectrum1D':
            return WaveSpectrum1D(**dictionary['data'])
        elif dictionary['__class__'] == 'WaveSpectrum2D':
            return WaveSpectrum2D(**dictionary['data'])
        elif dictionary['__class__'] == 'ndarray':
            return _b64_decode_numpy(dictionary['data'])
        elif dictionary['__class__'] == 'datetime':
            return datetime.fromisoformat( dictionary['data'] )
        elif dictionary['__class__'] == 'WaveBulkData':
            return WaveBulkData(**dictionary['data'])
        elif dictionary['__class__'] == 'WaveBulkData':
            return WaveBulkData(**dictionary['data'])
        elif dictionary['__class__'] == 'WindData':
            return WindData(**dictionary['data'])
        elif dictionary['__class__'] == 'SSTData':
            return SSTData(**dictionary['data'])
        elif dictionary['__class__'] == 'BarometricPressure':
            return BarometricPressure(**dictionary['data'])
    else:
        # legacy - should delete soonish.
        if 'directions' in dictionary:
            if isinstance(dictionary['directions'],numpy.ndarray):
                return dictionary
            # if directions -> create a wavespectrum2d. The keys directly map
            # to the constructor arguments.
            return WaveSpectrum2D(**dictionary)
        elif 'frequency' in dictionary:
            if isinstance(dictionary['frequency'],numpy.ndarray):
                return dictionary
            # if not directions but has frequency -> create a wavespectrum1d.
            # The keys directly map to the constructor arguments.
            return WaveSpectrum1D(**dictionary)
        elif 'dataframe' in dictionary:
            df = read_json(dictionary['dataframe'])
            df['timestamp'] = df['timestamp'].apply(lambda x: x.tz_localize('utc'))
            return df #DataFrame.from_dict(data['dataframe'])
        return dictionary

def load(filename: str) -> _UNION:
    """
    Load spectral data as saved by "save_spectrum" from the given file and
    return a (nested) object. The precise format of the output depends on what
    was saved.

    :param filename: path to file to load.
    :return:
        - Data in the same form it was saved.
    """

    with gzip.open(filename, 'rb') as file:
        data = file.read().decode('utf-8')
        return json.loads(data, object_hook=object_hook)


def save(_input: _UNION, filename: str, format='json', separate_spotters_into_files=False):
    """
    Save spectral data, possible in nested form as returned by the spectral
    partition/reconstruction/etc. functions. Data is saved in JSON form
    compressed with gzip.

    :param _input:
        - Data containing python primitives (dict/list) or any of the RogueWave
          classes.

    :param filename: path to save data.
    :return: None
    """

    def write(filename, _input, format):
        if format == 'json':
            with gzip.open(filename, 'wt') as file:
                json.dump(_input,file,cls=NumpyEncoder)
                #file.write(data.encode('utf-8'))
        else:
            raise Exception('Unknown output format')


    if separate_spotters_into_files:
        os.makedirs(filename, exist_ok=True)
        for key in _input:
            name = os.path.join(filename, key)
            write( name,_input[key],format )

    else:
        write(filename,_input,format)