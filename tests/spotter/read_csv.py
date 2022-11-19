from roguewave.spotter import (
    read_gps,
    read_displacement,
    read_location,
    read_spectra,
    read_raw_spectra,
)
from roguewave import FrequencySpectrum

PATH = "/Users/pietersmit/Downloads/Sunflower13/log"


def tst_read_gps(path=PATH):
    return read_gps(path)


def tst_read_displacement(path=PATH):
    displacement = read_displacement(path)

    assert "x" in displacement, "x is not displacement"
    assert "y" in displacement, "y is not displacement"
    assert "z" in displacement, "z is not displacement"
    assert "time" in displacement, "time is not displacement"
    assert displacement["x"][5] == 0.1824284208824179


def tst_read_spectra(path=PATH):
    spectra = read_spectra(path=path)
    assert isinstance(spectra, FrequencySpectrum)


def tst_read_location(path=PATH):
    location = read_location(path)
    assert "latitude" in location
    assert "longitude" in location
    assert "time" in location


def tst_read_gps_doppler_velocities(path=PATH):
    doppler_velocities = read_gps(path=path)
    assert "u" in doppler_velocities
    assert "v" in doppler_velocities
    assert "w" in doppler_velocities
    assert "time" in doppler_velocities


def tst_read_raw_spectra(path=PATH):
    return read_raw_spectra(path=path)


if __name__ == "__main__":
    tst_read_gps()
    tst_read_displacement()
    tst_read_gps_doppler_velocities()
    tst_read_location()
    tst_read_raw_spectra()
    tst_read_spectra()
