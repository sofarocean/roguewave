from tests.restart_files import clone_remote, bytes_hash
from roguewave import FrequencyDirectionSpectrum
from datetime import datetime, timezone
import numpy


def test_coordinates():
    restart_file = clone_remote()
    assert restart_file.coordinates(10) == (-78.0, 183.0)
    lat, lon = restart_file.coordinates(slice(11, 13, 1))
    assert lat[1] == -78.0
    assert lon[1] == 184.0


def test_direction():
    restart_file = clone_remote()
    for index, direction in enumerate(restart_file.direction):
        assert numpy.abs(direction - (index * 10 + 5)) < 1e-3


def test_frequency():
    restart_file = clone_remote()
    fstart = 0.035
    growth_factor = restart_file.grid._growth_factor

    for index, frequency in enumerate(restart_file.frequency):
        assert numpy.abs(fstart * growth_factor**index - frequency) < 1e-4


def test_header_bytes():
    restart_file = clone_remote()
    assert bytes_hash(restart_file.header_bytes()) == "d8fb87f5d4516d0ddcc8668bf0343487"


def test_interpolate_in_space():
    restart_file = clone_remote()
    lats = numpy.array((-1, -10))
    lons = numpy.array((-0.25, 359.9))
    spectra = restart_file.interpolate_in_space(lats, lons)

    assert isinstance(spectra, FrequencyDirectionSpectrum)
    assert len(spectra) == 2

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    assert numpy.all(
        numpy.abs(spectra.hm0() - numpy.array([1.9912376, 2.342316])) < 1e-3
    )


def test_latitude():
    restart_file = clone_remote()
    assert numpy.all(
        numpy.abs(
            restart_file.latitude
            - numpy.linspace(-90, 90, restart_file.number_of_latitudes, endpoint=True)
        )
        < 1.0e-3
    )


def test_longitude():
    restart_file = clone_remote()
    assert numpy.all(
        numpy.abs(
            restart_file.longitude
            - numpy.linspace(0, 360, restart_file.number_of_longitudes, endpoint=False)
        )
        < 1.0e-3
    )


def test_linear_indices():
    restart_file = clone_remote()
    assert numpy.all(
        numpy.abs(
            numpy.arange(restart_file.number_of_spatial_points)
            - restart_file.linear_indices
        )
        == 0
    )


def test_get_item():
    restart_file = clone_remote()

    # Normal indices
    spectra = restart_file[20000:20003]
    lats, lons = restart_file.coordinates(slice(20000, 20003, 1))

    assert isinstance(spectra, FrequencyDirectionSpectrum)
    assert len(spectra) == 3

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38909043, 3.31185516, 3.25957171])
    assert numpy.all(numpy.abs(spectra.hm0() - hm0) < 1e-3)

    # Fancy indices
    spectra = restart_file[[20000, 20001, 20002]]
    lats, lons = restart_file.coordinates([20000, 20001, 20002])

    assert isinstance(spectra, FrequencyDirectionSpectrum)
    assert len(spectra) == 3

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38909043, 3.31185516, 3.25957171])
    assert numpy.all(numpy.abs(spectra.hm0() - hm0) < 1e-3)

    # Scalar index
    spectra = restart_file[20000]
    lats, lons = restart_file.coordinates(20000)

    assert isinstance(spectra, FrequencyDirectionSpectrum)
    assert len(spectra) == 1

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38909043])
    assert numpy.all(numpy.abs(spectra.hm0() - hm0) < 1e-3)


def test_number_of_directions():
    restart_file = clone_remote()
    assert restart_file.number_of_directions == 36, restart_file.number_of_directions


def test_number_of_frequencies():
    restart_file = clone_remote()
    assert restart_file.number_of_frequencies == 36, restart_file.number_of_frequencies


def test_number_of_latitudes():
    restart_file = clone_remote()
    assert restart_file.number_of_latitudes == 361, restart_file.number_of_latitudes


def test_number_of_longitudes():
    restart_file = clone_remote()
    assert restart_file.number_of_longitudes == 720, restart_file.number_of_longitudes


def test_number_of_header_bytes():
    restart_file = clone_remote()
    assert (
        restart_file.number_of_header_bytes == 10368
    ), restart_file.number_of_header_bytes


def test_number_of_tail_bytes():
    restart_file = clone_remote()
    assert (
        restart_file.number_of_tail_bytes == 5640192
    ), restart_file.number_of_tail_bytes


def test_number_of_spatial_points():
    restart_file = clone_remote()
    assert (
        restart_file.number_of_spatial_points == 156635
    ), restart_file.number_of_spatial_points


def test_size_in_bytes():
    restart_file = clone_remote()
    assert restart_file.size_in_bytes == 817646400, restart_file.size_in_bytes


def test_number_of_spectral_points():
    restart_file = clone_remote()
    assert (
        restart_file.number_of_spectral_points == 1296
    ), restart_file.number_of_spectral_points


def test_time():
    restart_file = clone_remote()
    assert restart_file.time == datetime(
        2021, 6, 1, 6, 0, 0, tzinfo=timezone.utc
    ), restart_file.time


def test_variance():
    restart_file = clone_remote()
    m0 = restart_file.variance(slice(1, 112), slice(1, 111))
    assert m0.shape == (111, 110)


if __name__ == "__main__":
    test_number_of_header_bytes()
    test_number_of_tail_bytes()
    test_number_of_spatial_points()
    test_number_of_directions()
    test_number_of_frequencies()
    test_number_of_latitudes()
    test_number_of_longitudes()
    test_number_of_spectral_points()
    test_time()
    test_size_in_bytes()
    test_coordinates()
    test_direction()
    test_frequency()
    test_header_bytes()
    test_interpolate_in_space()
    test_latitude()
    test_longitude()
    test_linear_indices()
    test_get_item()
    test_variance()
