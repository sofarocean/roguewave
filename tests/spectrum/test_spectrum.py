import numpy as np

from roguewave import (
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
    load_spectrum_from_netcdf,
    concatenate_spectra,
)
from roguewave.wavespectra.parametric import create_parametric_spectrum
from roguewave.wavetheory.lineardispersion import GRAV
from roguewave.tools.time import to_datetime64
from numpy import linspace, inf, ndarray, pi, array, ones, nan, sqrt
from numpy.testing import assert_allclose
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
from xarray import DataArray
import os


def helper_create_spectrum() -> FrequencyDirectionSpectrum:
    angles = helper_angles()
    frequency = helper_frequency()
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


def helper_depth(N, d=inf):
    return ones(N) * d


def helper_frequency():
    return linspace(0.01, 1, 100, endpoint=True)


def helper_angles():
    return linspace(0, 360, 36, endpoint=False)


def helper_waveheights(N):
    return linspace(2, 4, N, endpoint=True)


def helper_latitude(N):
    return array([11.0 - ii * 0.01 for ii in range(0, N)])


def helper_longitude(N):
    return array([10 + ii * 0.01 for ii in range(0, N)])


def helper_time(N):
    return [
        datetime(2022, 10, 1, 6, 0, 0, tzinfo=timezone.utc) + ii * timedelta(hours=1)
        for ii in range(0, N)
    ]


def helper_create_spectra_list(N, depth=inf) -> List[FrequencyDirectionSpectrum]:
    """
    Helper to create a list of spectra
    :return:
    """
    angles = helper_angles()
    frequency = helper_frequency()
    waveheights = helper_waveheights(N)
    latitude = helper_latitude(N)
    longitude = helper_longitude(N)
    time = helper_time(N)
    depth = helper_depth(N, depth)

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
                longitude=longitude[ii],
                latitude=latitude[ii],
                time=time[ii],
                depth=depth[ii],
            )
        )
    return out


def helper_create_spectra(
    N, depth=inf
) -> Tuple[FrequencyDirectionSpectrum, FrequencySpectrum]:
    spectra = helper_create_spectra_list(N=N, depth=depth)
    spectra = concatenate_spectra(spectra, dim="time")
    return spectra, spectra.as_frequency_spectrum()


def test_concatenate():
    N = 4
    spectra = helper_create_spectra_list(N=N)

    spectrum = concatenate_spectra(spectra, dim="time")
    assert spectrum.variance_density.shape[0] == N
    assert list(spectrum.variance_density.dims) == ["time", "frequency", "direction"]
    assert len(spectrum.time) == N
    assert len(spectrum.depth) == N
    assert len(spectrum.latitude) == N
    assert len(spectrum.longitude) == N

    spectrum = concatenate_spectra(spectra, dim="latitude")
    assert spectrum.variance_density.shape[0] == N
    assert list(spectrum.variance_density.dims) == [
        "latitude",
        "frequency",
        "direction",
    ]
    assert spectrum.significant_waveheight.dims[0] == "latitude"
    assert_allclose(spectrum.significant_waveheight[-1], 4, 1e-3, 1e-3)


def test_concatenate_1d():
    N = 4
    spectra = [x.as_frequency_spectrum() for x in helper_create_spectra_list(N=N)]

    spectrum = concatenate_spectra(spectra, dim="time")
    assert spectrum.variance_density.shape[0] == N
    assert list(spectrum.variance_density.dims) == ["time", "frequency"]
    assert list(spectrum.a1.dims) == ["time", "frequency"]
    assert list(spectrum.b1.dims) == ["time", "frequency"]
    assert list(spectrum.a2.dims) == ["time", "frequency"]
    assert list(spectrum.b2.dims) == ["time", "frequency"]
    assert list(spectrum.time.dims) == ["time"]
    assert list(spectrum.depth.dims) == ["time"]
    assert list(spectrum.latitude.dims) == ["time"]
    assert list(spectrum.longitude.dims) == ["time"]
    assert list(spectrum.time.dims) == ["time"]
    assert len(spectrum.time) == N
    assert len(spectrum.depth) == N
    assert len(spectrum.latitude) == N
    assert len(spectrum.longitude) == N

    spectrum = concatenate_spectra(spectra, dim="latitude")
    assert spectrum.variance_density.shape[0] == N
    assert list(spectrum.variance_density.dims) == ["latitude", "frequency"]
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
    specs = helper_create_spectra(4)

    for spec in specs:
        mean = spec.mean(dim="time")
        assert len(mean) == 1, type(spec)


def test_spectrum1d():
    spec = helper_create_spectrum()
    spec1d = spec.as_frequency_spectrum()
    assert spec1d.dims_spectral == ["frequency"]
    assert spec1d.dims == ["frequency"]


def test_spectrum2d():
    (_, spec1d) = helper_create_spectra(4)

    for method in ["mem2", "mem"]:
        for solution_method in ["scipy", "newton", "approximate"]:
            spec2d = spec1d.as_frequency_direction_spectrum(
                72, method=method, solution_method=solution_method
            )

            assert spec2d.dims_spectral == ["frequency", "direction"]
            assert spec2d.dims == ["time", "frequency", "direction"]
            assert_allclose(
                spec2d.significant_waveheight,
                spec1d.significant_waveheight,
                rtol=1e-4,
                atol=1e-4,
            )
            assert_allclose(
                spec2d.mean_direction(), spec1d.mean_direction(), rtol=1e-1, atol=1e-1
            )


def test_sum():
    specs = helper_create_spectra(4)

    for spec in specs:
        _sum = spec.sum(dim="time")
        assert len(_sum) == 1, type(spec)


#
#
def test_std():
    specs = helper_create_spectra(4)

    for spec in specs:
        std = spec.std(dim="time")
        assert len(std) == 1, type(spec)
        assert type(spec) == type(std)


def helper_assert(
    dataarray: DataArray,
    dims: List[str],
    shape,
    values: ndarray = None,
    atol=1e-4,
    rtol=1e-4,
):
    assert list(dataarray.dims) == dims, f"{list(dataarray.dims)}, {dims}"
    assert dataarray.shape == shape, f"{list(dataarray.shape)}, {shape}"
    if values is not None:
        assert_allclose(values, dataarray.values, atol=atol, rtol=rtol)


def test_frequency_moment():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.frequency_moment(2), ["time"], (4,))


def test_number_of_frequencies():
    specs = helper_create_spectra(4)
    for spec in specs:
        assert spec.number_of_frequencies == len(helper_frequency())


def test_spectral_values():
    specs = helper_create_spectra(4)
    for spec in specs:

        if isinstance(spec, FrequencySpectrum):
            helper_assert(
                spec.spectral_values,
                ["time", "frequency"],
                (4, len(helper_frequency())),
            )
        else:
            helper_assert(
                spec.spectral_values,
                ["time", "frequency", "direction"],
                (4, len(helper_frequency()), len(helper_angles())),
            )


def test_radian_frequency():
    specs = helper_create_spectra(4)
    rad_freq = helper_frequency() * 2 * pi
    for spec in specs:
        helper_assert(spec.radian_frequency, ["frequency"], (len(rad_freq),), rad_freq)


def test_frequency():
    specs = helper_create_spectra(4)
    freq = helper_frequency()
    for spec in specs:
        helper_assert(spec.frequency, ["frequency"], (len(freq),), freq)


def test_latitude():
    specs = helper_create_spectra(5)
    lat = helper_latitude(5)
    for spec in specs:
        helper_assert(spec.latitude, ["time"], (len(lat),), lat)


def test_longitude():
    specs = helper_create_spectra(6)
    lon = helper_longitude(6)
    for spec in specs:
        helper_assert(spec.longitude, ["time"], (len(lon),), lon)


def test_time():
    specs = helper_create_spectra(6)
    time = to_datetime64(helper_time(6))
    for spec in specs:
        helper_assert(spec.time, ["time"], (len(time),), None)
        for t1, t2 in zip(spec.time, time):
            assert t1 == t2, (t1, t2)


def test_variance_density():
    specs = helper_create_spectra(4)
    for spec in specs:

        if isinstance(spec, FrequencySpectrum):
            helper_assert(
                spec.variance_density,
                ["time", "frequency"],
                (4, len(helper_frequency())),
            )
        else:
            helper_assert(
                spec.variance_density,
                ["time", "frequency", "direction"],
                (4, len(helper_frequency()), len(helper_angles())),
            )


def test_e():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.e, ["time", "frequency"], (4, len(helper_frequency())))


def test_a1():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.a1, ["time", "frequency"], (4, len(helper_frequency())))


def test_b1():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.b1, ["time", "frequency"], (4, len(helper_frequency())))


def test_a2():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.a2, ["time", "frequency"], (4, len(helper_frequency())))


def test_b2():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.b2, ["time", "frequency"], (4, len(helper_frequency())))


def test_m0():
    specs = helper_create_spectra(4)
    m0 = helper_waveheights(4) ** 2 / 16
    for spec in specs:
        helper_assert(spec.m0(), ["time"], (4,), m0)
        helper_assert(spec.m0(), ["time"], (4,), spec.frequency_moment(0).values)


def test_m1():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.m1(), ["time"], (4,), spec.frequency_moment(1).values)


def test_m2():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.m2(), ["time"], (4,), spec.frequency_moment(2).values)


def test_hm0():
    specs = helper_create_spectra(4)
    hm0 = helper_waveheights(4)
    for spec in specs:
        helper_assert(spec.hm0(), ["time"], (4,), hm0)

def test_hm0_partial():
    specs = helper_create_spectra(4)
    hm0 = helper_waveheights(4)
    for spec in specs:
        hm0low = spec.hm0(fmax=0.2)
        hm0high = spec.hm0(fmin=0.2)
        hm0_total = np.sqrt(hm0low**2 + hm0high**2)
        helper_assert(hm0_total, ["time"], (4,), hm0)

        hm0low = spec.hm0(fmax=0.201)
        hm0high = spec.hm0(fmin=0.201)
        hm0_total = np.sqrt(hm0low**2 + hm0high**2)
        helper_assert(hm0_total, ["time"], (4,), hm0)


def test_tm01():
    specs = helper_create_spectra(4)

    tm01 = array([7.72671553, 7.72671553, 7.72671553, 7.72671553])
    for spec in specs:
        helper_assert(spec.tm01(), ["time"], (4,), tm01)


def test_tm02():
    specs = helper_create_spectra(4)
    tm02 = array([7.14851571, 7.14851571, 7.14851571, 7.14851571])
    for spec in specs:
        helper_assert(spec.tm02(), ["time"], (4,), tm02)


def test_peak_index():
    specs = helper_create_spectra(4)
    ipeak = array([9, 9, 9, 9])
    for spec in specs:
        helper_assert(spec.peak_index(), ["time"], (4,), ipeak)


def test_peak_frequency():
    specs = helper_create_spectra(4)
    fpeak = array([0.1, 0.1, 0.1, 0.1])
    for spec in specs:
        helper_assert(spec.peak_frequency(), ["time"], (4,), fpeak)


def test_peak_period():
    specs = helper_create_spectra(4)
    peak_period = 1 / array([0.1, 0.1, 0.1, 0.1])
    for spec in specs:
        helper_assert(spec.peak_period(), ["time"], (4,), peak_period)


def test_peak_direction():
    specs = helper_create_spectra(4)
    peak_direction = array([20, 20, 20, 20])
    for spec in specs:
        helper_assert(spec.peak_direction(), ["time"], (4,), peak_direction)


def test_peak_directional_spread():
    specs = helper_create_spectra(4)
    peak_directional_spread = array([10, 10, 10, 10])
    for spec in specs:
        helper_assert(
            spec.peak_directional_spread(),
            ["time"],
            (4,),
            peak_directional_spread,
            atol=0.2,
            rtol=0.2,
        )


def test_mean_direction():
    specs = helper_create_spectra(4)
    mean_direction = array([20, 20, 20, 20])
    for spec in specs:
        helper_assert(spec.mean_direction(), ["time"], (4,), mean_direction)


def test_mean_direction_per_frequency():
    specs = helper_create_spectra(4)
    dir_per_freq = ones((4, len(helper_frequency()))) * 20
    dir_per_freq[:, 0:2] = nan
    for spec in specs:
        helper_assert(
            spec.mean_direction_per_frequency,
            ["time", "frequency"],
            (4, 100),
            dir_per_freq,
        )


def test_mean_spread_per_frequency():
    specs = helper_create_spectra(4)
    spread_per_freq = ones((4, len(helper_frequency()))) * 10
    spread_per_freq[:, 0:2] = nan
    for spec in specs:
        helper_assert(
            spec.mean_spread_per_frequency,
            ["time", "frequency"],
            (4, 100),
            spread_per_freq,
            rtol=0.2,
            atol=0.2,
        )


def test_mean_directional_spread():
    specs = helper_create_spectra(4)
    mean_directional_spread = array([10, 10, 10, 10])
    for spec in specs:
        helper_assert(
            spec.mean_directional_spread(),
            ["time"],
            (4,),
            mean_directional_spread,
            rtol=0.2,
            atol=0.2,
        )


def test_mean_a1():
    specs = helper_create_spectra(4)
    mean_a1 = array([0.925048, 0.925048, 0.925048, 0.925048])
    for spec in specs:
        helper_assert(spec.mean_a1(), ["time"], (4,), mean_a1)


def test_mean_b1():
    specs = helper_create_spectra(4)
    mean_b1 = array([0.33669, 0.33669, 0.33669, 0.33669])
    for spec in specs:
        helper_assert(spec.mean_b1(), ["time"], (4,), mean_b1)


def test_mean_a2():
    specs = helper_create_spectra(4)
    mean_a2 = array([0.719374, 0.719374, 0.719374, 0.719374])
    for spec in specs:
        helper_assert(spec.mean_a2(), ["time"], (4,), mean_a2)


def test_mean_b2():
    specs = helper_create_spectra(4)
    mean_b2 = array([0.603627, 0.603627, 0.603627, 0.603627])
    for spec in specs:
        helper_assert(spec.mean_b2(), ["time"], (4,), mean_b2)


def test_depth():
    depth = 100
    specs = helper_create_spectra(4, depth=depth)
    depth = helper_depth(4, depth)
    for spec in specs:
        helper_assert(spec.depth, ["time"], (4,), depth)


def test_wavenumber():
    depth = 0.00001
    frequency = helper_frequency()
    specs = helper_create_spectra(4, depth=depth)
    depth = helper_depth(4, depth)
    wavenumber_shallow = frequency[None, :] * pi * 2 / sqrt(GRAV * depth[:, None])
    for spec in specs:
        helper_assert(
            spec.wavenumber,
            ["time", "frequency"],
            (4, len(frequency)),
            wavenumber_shallow,
        )

    depth = inf
    frequency = helper_frequency()
    specs = helper_create_spectra(4, depth=depth)
    depth = helper_depth(4, depth)
    wavenumber_deep = (
        (frequency[None, :] * pi * 2) ** 2 / GRAV * ones((4, len(frequency)))
    )
    for spec in specs:
        helper_assert(
            spec.wavenumber, ["time", "frequency"], (4, len(frequency)), wavenumber_deep
        )


def test_wavelength():
    depth = 0.00001
    frequency = helper_frequency()
    specs = helper_create_spectra(4, depth=depth)
    depth = helper_depth(4, depth)
    wavelength_shallow = sqrt(GRAV * depth[:, None]) / frequency[None, :]
    for spec in specs:
        helper_assert(
            spec.wavelength,
            ["time", "frequency"],
            (4, len(frequency)),
            wavelength_shallow,
        )


def test_peak_wavenumber():
    specs = helper_create_spectra(4)
    for spec in specs:
        ipeak = spec.peak_index()
        peak_wavenumber = spec.wavenumber[:, ipeak].values
        helper_assert(spec.peak_wavenumber, ["time"], (4,), peak_wavenumber)


def test_bulk_variables():
    specs = helper_create_spectra(4)

    bulk_vars = [
        "significant_waveheight",
        "mean_period",
        "peak_period",
        "peak_direction",
        "peak_directional_spread",
        "mean_direction",
        "mean_directional_spread",
        "peak_frequency",
        "latitude",
        "longitude",
        "timestamp",
    ]
    for spec in specs:
        bulk = spec.bulk_variables()
        for var in bulk_vars:
            assert var in bulk

        for var in bulk:
            helper_assert(bulk[var], ["time"], (4,), None)


def test_significant_waveheight():
    specs = helper_create_spectra(4)
    for spec in specs:
        hm0 = spec.hm0().values
        helper_assert(spec.significant_waveheight, ["time"], (4,), hm0)


def test_mean_period():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.mean_period, ["time"], (4,), spec.tm01().values)


def test_zero_crossing_period():
    specs = helper_create_spectra(4)
    for spec in specs:
        helper_assert(spec.zero_crossing_period, ["time"], (4,), spec.tm02().values)


def test_interpolate():
    specs = helper_create_spectra(4)
    time = helper_time(4)
    intp_time = time[2] + timedelta(minutes=30)
    hm0 = helper_waveheights(4)
    m0 = hm0**2 / 16
    intp_hm0 = 4 * sqrt((m0[2] + m0[3]) / 2)

    for spec in specs:
        intp_spec = spec.interpolate({"time": intp_time})
        assert_allclose(intp_spec.hm0(), intp_hm0, rtol=1e-3, atol=1e-3)
        assert intp_spec.dims == spec.dims


def test_interpolate_frequency():
    specs = helper_create_spectra(4)
    freq = linspace(0.0, 2, 1000)

    for spec in specs:
        intp_spec = spec.interpolate_frequency(freq)
        assert intp_spec.dims == spec.dims
        assert_allclose(intp_spec.hm0(), spec.hm0(), rtol=1e-3, atol=1e-3)
        assert_allclose(
            intp_spec.mean_direction(), spec.mean_direction(), rtol=1e-3, atol=1e-3
        )

    spec = specs[1]

    intp_spec = spec.interpolate_frequency(
        freq, method="spline", monotone_interpolation=True
    )
    assert intp_spec.dims == spec.dims
    assert_allclose(intp_spec.hm0(), spec.hm0(), rtol=1e-3, atol=1e-3)
    assert_allclose(
        intp_spec.mean_direction(), spec.mean_direction(), rtol=1e-3, atol=1e-3
    )

    intp_spec = spec.interpolate_frequency(
        freq, method="spline", monotone_interpolation=False
    )
    assert intp_spec.dims == spec.dims
    assert_allclose(intp_spec.hm0(), spec.hm0(), rtol=1e-3, atol=1e-3)
    assert_allclose(
        intp_spec.mean_direction(), spec.mean_direction(), rtol=1e-3, atol=1e-3
    )

    intp_spec = spec.interpolate_frequency(
        freq, method="nearest", monotone_interpolation=True
    )
    assert intp_spec.dims == spec.dims
    assert_allclose(intp_spec.hm0(), spec.hm0(), rtol=1e-3, atol=1e-3)
    assert_allclose(
        intp_spec.mean_direction(), spec.mean_direction(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_interpolate_frequency()
    test_concatenate()
    test_spectrum1d()
    test_concatenate_1d()
    test_spectrum2d()
    test_sel()
    test_isel()
    test_save_and_load()
    test_mean()
    test_sum()
    test___get_item__()
    test_frequency_moment()
    test_number_of_frequencies()
    test_spectral_values()
    test_radian_frequency()
    test_frequency()
    test_latitude()
    test_longitude()
    test_time()
    test_variance_density()
    test_e()
    test_a1()
    test_b1()
    test_a2()
    test_b2()
    test_m0()
    test_m1()
    test_m2()
    test_tm01()
    test_tm02()
    test_peak_index()
    test_peak_frequency()
    test_peak_period()
    test_peak_direction()
    test_peak_directional_spread()
    test_mean_direction()
    test_mean_direction_per_frequency()
    test_mean_spread_per_frequency()
    test_mean_directional_spread()
    test_mean_a1()
    test_mean_b1()
    test_mean_a2()
    test_mean_b2()
    test_depth()
    test_wavenumber()
    test_wavelength()
    test_peak_wavenumber()
    test_bulk_variables()
    test_significant_waveheight()
    test_mean_period()
    test_zero_crossing_period()
    test_interpolate()
