import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .estimators import spec2d_from_spec1d, spec1d_from_spec2d

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())

def spectrum1D( frequency , varianceDensity ,a1=None,b1=None,a2=None,b2=None, latitude=None, longitude=None, timestamp=None)->WaveSpectrum1D:
    input = WaveSpectrum1DInput(
        frequency=frequency,
        varianceDensity=varianceDensity,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        a1=a1,
        b1=b1,
        a2=a2,
        b2=b2
    )
    return WaveSpectrum1D(input)


def spectrum2D( frequency , directions, varianceDensity, latitude=None, longitude=None, timestamp=None)->WaveSpectrum2D:
    input = WaveSpectrum2DInput(
        frequency=frequency,
        varianceDensity=varianceDensity,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        directions=directions
    )
    return WaveSpectrum2D(input)