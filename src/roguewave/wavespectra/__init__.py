from .spectrum2D import WaveSpectrum2D
from .spectrum1D import WaveSpectrum1D
from .estimators import spec2d_from_spec1d, spec1d_from_spec2d
from xarray import DataArray, Dataset
from typing import List, Union, Sequence
from roguewave.tools.time import to_datetime64
import numpy
from roguewave import logger

def wave_spectra_as_data_set(
        spectra:Union[Sequence[WaveSpectrum1D],Sequence[WaveSpectrum2D]]
) -> Dataset:

    if isinstance(spectra[0],WaveSpectrum1D ):
        spectra = [ spec2d_from_spec1d(x) for x in spectra ]

    nfreq = len(spectra[0].frequency)
    ndir  = len(spectra[0].direction)
    nt = len(spectra)
    data = numpy.empty( (nt, nfreq, ndir ), dtype='float32' )

    time = to_datetime64([x.timestamp for x in spectra])
    for it,spec in enumerate(spectra):
        data[it,:,:] = spectra[it].variance_density

    return Dataset(
        data_vars={
            'variance_density':(('time', 'frequency','direction'),data  )
        },
        coords={
            'time': time,
            'frequency':spectra[0].frequency,
            'direction':spectra[0].direction
        }
    )

