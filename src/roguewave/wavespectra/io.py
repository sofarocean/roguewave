from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D
from .wavespectrum import WaveSpectrum
from . import spectrum1D, spectrum2D
import gzip
import json


def load_spectrum( filename ):
    with gzip.open(filename,'rb') as file:
        data = file.read().decode('utf-8')

    data = _create_object_from_json(json.loads(data))
    return data


def _create_object_from_json(data):
    if isinstance(data,(dict)):
        if 'directions' in data:
            return spectrum2D(**data)
        elif 'frequency' in data:
            return spectrum1D(**data)
        else:
            output = {}
            for key in data:
                output[key] = _create_object_from_json(data[key])
            return output
    elif isinstance(data,list):
        output = []
        for item in data:
            output.append(_create_object_from_json(item))
        return output
    else:
        raise Exception('Cannot convert to json compatible type')


def _create_jsonable_object(_input):
    if isinstance(_input, WaveSpectrum):
        if isinstance(_input, WaveSpectrum1D):
            return _input._create_wave_spectrum_input()
        elif isinstance(_input, WaveSpectrum2D):
            return _input._create_wave_spectrum_input()
    elif isinstance(_input,(dict)):
        output = {}
        for key in _input:
            output[key] = _create_jsonable_object(_input[key])
        return output
    elif isinstance(_input,list):
        output = []
        for item in _input:
            output.append(_create_jsonable_object(item))
        return output
    else:
        raise Exception('Cannot convert to json compatible type')


def save_spectrum( _input, filename ):
    output = _create_jsonable_object(_input)

    with gzip.open(filename,'wb') as file:
        data = json.dumps(output)
        file.write(data.encode('utf-8'))