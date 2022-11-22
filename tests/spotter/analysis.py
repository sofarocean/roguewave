from roguewave.spotter import (
    displacement_from_gps_doppler_velocities,
    displacement_from_gps_positions,
    spectra_from_displacement,
    spectra_from_raw_gps,
)
from roguewave import FrequencySpectrum
from roguewave.spotter import read_displacement
from numpy import interp
from numpy.testing import assert_allclose
from roguewave.tools.time import datetime64_to_timestamp

PATH = "/Users/pietersmit/Desktop/tmp"  # "/Users/pietersmit/Downloads/Sunflower13/log"


def tst_spectra_from_displacement(path=PATH):
    calc_spec = spectra_from_displacement(path)
    assert isinstance(calc_spec, FrequencySpectrum)


def tst_spectra_from_gps(path=PATH):
    calc_spec = spectra_from_raw_gps(path)
    assert isinstance(calc_spec, FrequencySpectrum)

    # t = datetime64_to_timestamp(calc_spec.time.values[40:140])
    # time_embed = datetime64_to_timestamp(em_spec.time.values)
    #
    # hs_embed = interp(t, time_embed, em_spec.hm0().values)
    # hs_calc = calc_spec.hm0().values[40:140]
    #
    # import matplotlib.pyplot as plt
    # plt.plot(calc_spec.time.values[40:140], hs_calc, 'k')
    # plt.plot(calc_spec.time.values[40:140], hs_embed, 'r')
    #
    # plt.figure()
    # plt.plot(hs_calc - hs_embed)
    # plt.show()


def tst_spectra_from_raw_gps(path=PATH):
    return spectra_from_raw_gps(path)


def tst_displacement_from_gps_positions(path=PATH):
    displacement = displacement_from_gps_positions(path=path)
    embedded_displacement = read_displacement(path=path)

    assert "time" in displacement
    assert "x" in displacement
    assert "y" in displacement
    assert "z" in displacement

    t = datetime64_to_timestamp(displacement["time"].values[1500:2000])
    x = displacement["x"].values[1500:2000]
    y = displacement["y"].values[1500:2000]

    time_embed = datetime64_to_timestamp(embedded_displacement["time"].values)

    xe = interp(t, time_embed, embedded_displacement["x"].values)
    ye = interp(t, time_embed, embedded_displacement["y"].values)

    # Test if displacements are close to embedded values. Note that this only holds for horizontal motions as
    # vertical motions are estimated from velocity on the embedded side.
    assert_allclose(x, xe, rtol=0.01, atol=0.01)
    assert_allclose(y, ye, rtol=0.01, atol=0.01)


def tst_displacement_from_gps_velocities(path=PATH):
    displacement = displacement_from_gps_doppler_velocities(path=path)
    embedded_displacement = read_displacement(path=path)

    assert "time" in displacement
    assert "x" in displacement
    assert "y" in displacement
    assert "z" in displacement

    t = displacement["time"].values.astype("float64")[1500:2000]
    z = displacement["z"].values[1500:2000]

    ze = interp(
        t,
        embedded_displacement["time"].values.astype("float64"),
        embedded_displacement["z"].values,
    )
    assert_allclose(z, ze, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    # tst_displacement_from_gps_positions(path=PATH)
    # tst_displacement_from_gps_velocities(path=PATH)
    # tst_spectra_from_displacement(path=PATH)
    tst_spectra_from_gps(path=PATH)
    # tst_embedded_spectra(path=PATH)
