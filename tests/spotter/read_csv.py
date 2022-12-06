from roguewave.spotter import (
    read_spectra,
    read_displacement,
    read_gps,
    read_raw_spectra,
    read_baro,
    read_raindb,
    read_baro_raw,
    read_sst,
    read_gmn,
    read_location,
    read_data,
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
    assert displacement["x"][5] == 0.1824284178119656, displacement["x"][5]


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


def tst_read_csv_data(path=PATH):
    return read_data(path, "BARO_RAW")


def tst_read_baro(path=PATH):
    return read_baro(path)


def tst_read_baro_raw(path=PATH):
    return read_baro_raw(path)


def tst_read_sst(path=PATH):
    return read_sst(path)


def tst_read_raindb(path=PATH):
    return read_raindb(path)


def tst_read_gmn(path=PATH):
    return read_gmn(path)


if __name__ == "__main__":
    tst_read_gps()
    tst_read_displacement()
    tst_read_gps_doppler_velocities()
    tst_read_location()
    tst_read_raw_spectra()
    tst_read_spectra()
    tst_read_csv_data()
    tst_read_gmn()
    tst_read_raindb()
    tst_read_sst()
    tst_read_baro_raw()
    tst_read_baro()
