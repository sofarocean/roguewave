import json
from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
import types
import numpy
from . import resources


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

def get_spectrum()->WaveSpectrum1D:
    test_spectrum = json.loads(pkg_resources.read_text(resources, 'spectrum.json'))
    return WaveSpectrum1D( WaveSpectrum1DInput(**test_spectrum) )

def test_methods_1d():
    test_data = json.loads(
        pkg_resources.read_text(resources, 'spectrum1D_data.json'))

    spectrum = get_spectrum()
    for key,value in test_data.items():
        func=getattr(spectrum,key)

        if type(func)==types.MethodType:
            result = func()
        else:
            result = func

        difference = numpy.sum(numpy.abs(result - value))

        error = f'Output of {key} from WaveSpectrum1D is {result} but should be {value}'
        assert difference == 0.0 , error

def test_methods_2d():
    test_data = json.loads(
        pkg_resources.read_text(resources, 'spectrum1D_data.json'))

    spectrum = get_spectrum().spectrum2d(36)
    for key,value in test_data.items():
        func=getattr(spectrum,key)

        if type(func)==types.MethodType:
            result = func()
        else:
            result = func

        difference = numpy.sum(numpy.abs(result - value))

        error = f'Output of {key} from WaveSpectrum2D is {result} but should be {value}, difference {difference}'
        assert difference <= 1.0e-6, error