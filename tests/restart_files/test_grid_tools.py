from tests.restart_files import clone_remote
from xarray import DataArray
from roguewavespectrum import Spectrum
from roguewave import unpack_ww3_data


# , concatenate_spectra

import numpy


def test_unpack_spectrum():
    restart_file = clone_remote()
    E = restart_file[:]
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, Spectrum)

    hsig = E.hm0()
    depth = E.depth
    assert hsig.shape == (361, 720)
    assert depth.shape == (361, 720)


def test_unpack_dataset():
    restart_file = clone_remote()
    E = restart_file[:].directional_variance_density
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, DataArray)
    assert E.shape == (361, 720, 36, 36)


def test_unpack_numpy():
    restart_file = clone_remote()
    E = restart_file[:].directional_variance_density.values
    E = unpack_ww3_data(E, restart_file.grid)
    assert isinstance(E, numpy.ndarray)
    assert E.shape == (361, 720, 36, 36)


if __name__ == "__main__":
    test_unpack_spectrum()
    test_unpack_numpy()
    test_unpack_dataset()
