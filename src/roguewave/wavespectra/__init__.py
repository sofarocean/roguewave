from .spectrum2D import WaveSpectrum2D
from .spectrum1D import WaveSpectrum1D
from .estimators import spec2d_from_spec1d, spec1d_from_spec2d
from roguewave import logger
from xarray import DataArray
from typing import List, Union, Sequence
import numpy


def wave_spectra_as_data_array(
        spectra:Union[Sequence[WaveSpectrum1D],Sequence[WaveSpectrum2D]]
) -> DataArray:

    if isinstance(spectra[0],WaveSpectrum1D ):
        spectra = [ spec2d_from_spec1d(x) for x in spectra ]

    nfreq = len(spectra[0].frequency)
    ndir  = len(spectra[0].direction)
    nt = len(spectra)
    data = numpy.empty( (nt, nfreq, ndir ), dtype='float32' )

    time = [x.timestamp for x in spectra]
    for it,spec in enumerate(spectra):
        data[it,:,:] = spectra[it].variance_density

    return DataArray(
        data,
        coords={'time': time, 'frequency':spectra[0].frequency,
                'direction':spectra[0].direction},
        dims=['time', 'frequency','direction'],
        name='variance density',
    )

