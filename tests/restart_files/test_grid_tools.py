from tests.restart_files import clone_remote
from xarray import DataArray
from roguewave import (
    FrequencyDirectionSpectrum,
    unpack_ww3_data,
)


# , concatenate_spectra

import numpy


def test_unpack_spectrum():
    restart_file = clone_remote()
    E = restart_file[:]
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, FrequencyDirectionSpectrum)

    hsig = E.significant_waveheight
    depth = E.depth
    assert hsig.shape == (361, 720)
    assert depth.shape == (361, 720)


def test_unpack_dataset():
    restart_file = clone_remote()
    E = restart_file[:].variance_density
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, DataArray)
    assert E.shape == (361, 720, 36, 36)


def test_unpack_numpy():
    restart_file = clone_remote()
    E = restart_file[:].variance_density.values
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, numpy.ndarray)
    assert E.shape == (361, 720, 36, 36)


# def concat():
#     restart_file = clone_remote()
#     restart_file2 = clone_remote()
#     E = [restart_file[:],restart_file2[:]]
#     d = concatenate_spectra(E,dim='ensemble')
#     d = unpack_ww3_data(d.significant_waveheight,restart_file.grid)
#     print(d.mean(dim='ensemble',skipna=False))
#     print(d.std(dim='ensemble',skipna=False))

if __name__ == "__main__":
    test_unpack_spectrum()
    test_unpack_numpy()
    test_unpack_dataset()
