from roguewave import (
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
    load_spectrum_from_netcdf,
    concatenate_spectra,
)
from roguewave.wavespectra.parametric import create_parametric_spectrum
from numpy import linspace, inf
from numpy.testing import assert_allclose
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import os


def helper_create_spectrum() -> FrequencyDirectionSpectrum:
    angles = linspace(0, 360, 36, endpoint=False)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    return create_parametric_spectrum(
        frequency_hertz=frequency,
        frequency_shape="pm",
        peak_frequency_hertz=0.1,
        significant_wave_height=2,
        direction_degrees=angles,
        direction_shape="raised_cosine",
        mean_direction_degrees=20,
        width_degrees=10,
        longitude=10,
        latitude=11,
        time=datetime(2022, 10, 1, 6, 0, 0, tzinfo=timezone.utc),
        depth=inf,
    )


def helper_create_spectra_list(N) -> List[FrequencyDirectionSpectrum]:
    """
    Helper to create a list of spectra
    :return:
    """
    angles = linspace(0, 360, 36, endpoint=False)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    waveheights = linspace(2, 4, N, endpoint=True)

    out = []
    for ii, wh in enumerate(waveheights):
        out.append(
            create_parametric_spectrum(
                frequency_hertz=frequency,
                frequency_shape="pm",
                peak_frequency_hertz=0.1,
                significant_wave_height=wh,
                direction_degrees=angles,
                direction_shape="raised_cosine",
                mean_direction_degrees=20,
                width_degrees=10,
                longitude=10 + ii * 0.01,
                latitude=11 - ii * 0.01,
                time=datetime(2022, 10, 1, 6, 0, 0, tzinfo=timezone.utc)
                + ii * timedelta(hours=1),
                depth=inf,
            )
        )
    return out


def helper_create_spectra(N) -> Tuple[FrequencyDirectionSpectrum, FrequencySpectrum]:
    spectra = helper_create_spectra_list(N=N)
    spectra = concatenate_spectra(spectra, dim="time")
    return spectra, spectra.spectrum_1d()


def test_concatenate():
    N = 4
    spectra = helper_create_spectra_list(N=N)

    spectrum = concatenate_spectra(spectra, dim="time")
    assert spectrum.variance_density.shape[0] == N
    assert spectrum.variance_density.dims[0] == "time"
    assert len(spectrum.time) == N
    assert len(spectrum.depth) == N
    assert len(spectrum.latitude) == N
    assert len(spectrum.longitude) == N

    spectrum = concatenate_spectra(spectra, dim="latitude")
    assert spectrum.variance_density.shape[0] == N
    assert spectrum.variance_density.dims[0] == "latitude"
    assert spectrum.significant_waveheight.dims[0] == "latitude"
    assert_allclose(spectrum.significant_waveheight[-1], 4, 1e-3, 1e-3)


def test_save_and_load():
    spec = helper_create_spectrum()
    spec.save_as_netcdf("test.nc")

    new_spec = load_spectrum_from_netcdf("test.nc")
    assert_allclose(spec.hm0(), new_spec.hm0(), 1e-4, 1e-4)
    os.remove("test.nc")

    spec = concatenate_spectra(helper_create_spectra_list(4), dim="time")
    spec.save_as_netcdf("test2.nc")

    new_spec = load_spectrum_from_netcdf("test2.nc")
    assert_allclose(spec.hm0(), new_spec.hm0(), 1e-4, 1e-4)
    os.remove("test2.nc")


def test_sel():
    (spec2d, spec1d) = helper_create_spectra(4)

    for spec in (spec2d, spec1d):
        time = [spec.time[1].values, spec.time[2].values]
        data = spec.sel(time=time)
        assert len(data) == 2
        assert data.time[0] == time[0]


def test_isel():
    (spec2d, spec1d) = helper_create_spectra(4)

    for spec in (spec2d, spec1d):
        time = [spec.time[1].values, spec.time[2].values]
        data = spec.isel(time=[1, 2])
        assert len(data) == 2
        assert data.time[0] == time[0]


def test___get_item__():
    (spec2d, spec1d) = helper_create_spectra(4)

    for spec in (spec2d, spec1d):
        time = [spec.time[1].values, spec.time[2].values]

        if isinstance(spec, FrequencySpectrum):
            data = spec[1:3, :]
        else:
            data = spec[1:3, :, :]

        assert len(data) == 2
        assert data.time[0] == time[0]


def test_mean():
    (spec2d, spec1d) = helper_create_spectra(4)

    for spec in (spec2d, spec1d):
        mean = spec.mean(dim="time")
        assert len(mean) == 1


# def test_sum():
#     spec = helper_create_spectra(4)
#
#
# def test_std():
#     spec = helper_create_spectra(4)
#
#
# def test_frequency_moment():
#     spec = helper_create_spectra(4)
#
#
# def test_number_of_frequencies():
#     spec = helper_create_spectra(4)
#
#
# def test_spectral_values():
#     spec = helper_create_spectra(4)
#
#
# def test_radian_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test_latitude():
#     spec = helper_create_spectra(4)
#
#
# def test_longitude():
#     spec = helper_create_spectra(4)
#
#
# def test_time():
#     spec = helper_create_spectra(4)
#
#
# def test_variance_density():
#     spec = helper_create_spectra(4)
#
#
# def test_e():
#     spec = helper_create_spectra(4)
#
#
# def test_a1():
#     spec = helper_create_spectra(4)
#
#
# def test_b1():
#     spec = helper_create_spectra(4)
#
#
# def test_a2():
#     spec = helper_create_spectra(4)
#
#
# def test_b2():
#     spec = helper_create_spectra(4)
#
#
# def test_a1():
#     spec = helper_create_spectra(4)
#
#
# def test_b1():
#     spec = helper_create_spectra(4)
#
#
# def test_a2():
#     spec = helper_create_spectra(4)
#
#
# def test_b2():
#     spec = helper_create_spectra(4)
#
#
# def test_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test_m0():
#     spec = helper_create_spectra(4)
#
#
# def test_m1():
#     spec = helper_create_spectra(4)
#
#
# def test_m2():
#     spec = helper_create_spectra(4)
#
#
# def test_hm0():
#     spec = helper_create_spectra(4)
#
#
# def test_tm01():
#     spec = helper_create_spectra(4)
#
#
# def test_tm02():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_index():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_period():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_direction():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_directional_spread():
#     spec = helper_create_spectra(4)
#
#
# def test__mean_direction():
#     spec = helper_create_spectra(4)
#
#
# def test__spread():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_direction_per_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_spread_per_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test__spectral_weighted():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_direction():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_directional_spread():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_a1():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_b1():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_a2():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_b2():
#     spec = helper_create_spectra(4)
#
#
# def test_depth():
#     spec = helper_create_spectra(4)
#
#
# def test_wavenumber():
#     spec = helper_create_spectra(4)
#
#
# def test_wavelength():
#     spec = helper_create_spectra(4)
#
#
# def test_peak_wavenumber():
#     spec = helper_create_spectra(4)
#
#
# def test_bulk_variables():
#     spec = helper_create_spectra(4)
#
#
# def test_significant_waveheight():
#     spec = helper_create_spectra(4)
#
#
# def test_mean_period():
#     spec = helper_create_spectra(4)
#
#
# def test_zero_crossing_period():
#     spec = helper_create_spectra(4)
#
#
# def test_interpolate():
#     spec = helper_create_spectra(4)
#
#
# def test_interpolate_frequency():
#     spec = helper_create_spectra(4)
#
#
# def test__range():
#     spec = helper_create_spectra(4)
#
#
# def test_save_as_netcdf():
#     spec = helper_create_spectra(4)


if __name__ == "__main__":
    test_sel()
    test_isel()
    test_concatenate()
    test_save_and_load()
    test_mean()
    test___get_item__()
