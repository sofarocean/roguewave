from tests.restart_files import clone_remote, bytes_hash
from roguewave import Spectrum
from roguewave.wavewatch3.grid_tools import Grid
from roguewave.wavewatch3.restart_file import RestartFile
from roguewave.wavewatch3.restart_file_metadata import MetaData
from datetime import datetime, timezone
import numpy


class _FakeResource:
    """
    Minimal resource stand-in: interpolate_in_space's spectral-data path
    reads through RestartFile.__getitem__/_fancy_index, which only needs
    read_range to return zero-filled records of the right size -- the
    spectral values themselves are irrelevant to this depth-only
    regression test.
    """

    def __init__(self, record_size_bytes):
        self._record_size_bytes = record_size_bytes

    def read_range(self, slices):
        return [bytes(self._record_size_bytes) for _ in slices]


def _make_two_cell_restart_file():
    """
    A synthetic, in-memory RestartFile with two stacked 2x2 lat/lon
    bilinear cells sharing a latitude row: cell A ((0,0)-(1,1)) is fully
    sea at depth=100m; cell B ((1,0)-(2,1)) has one masked/land corner at
    (2,1). Interpolating at both cells' centers in a single call exercises
    the real (not re-implemented) nested _get_depth closure while never
    sending an empty index array to _fancy_index for any one bilinear
    corner role -- every role has at least one valid point via cell A, so
    this doesn't depend on the unrelated empty-fancy-index bug (#19,
    already fixed on cg/nan_filter_restart_interpolation, out of scope
    here).
    """
    number_of_frequencies = 2
    number_of_directions = 2
    frequencies = numpy.array([0.1, 0.2])
    directions = numpy.array([0.0, 180.0])
    latitude = numpy.array([0.0, 1.0, 2.0])
    longitude = numpy.array([0.0, 1.0])

    # [ilat, ilon] -> linear index; (2, 1) is masked/land.
    to_linear_index = numpy.array([[0, 1], [2, 3], [4, -1]])
    # linear index -> [ilon, ilat]
    to_point_index = numpy.array([[0, 1, 0, 1, 0], [0, 0, 1, 1, 2]])

    grid = Grid(
        number_of_spatial_points=5,
        frequencies=frequencies,
        directions=directions,
        latitude=latitude,
        longitude=longitude,
        depth=numpy.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        mask=numpy.array([[1, 1], [1, 1], [1, 0]]),
        _to_linear_index=to_linear_index,
        _to_point_index=to_point_index,
    )
    meta_data = MetaData(
        name="test",
        version="1",
        grid_name="test",
        restart_type="test",
        nsea=5,
        nspec=number_of_frequencies * number_of_directions,
        record_size_bytes=number_of_frequencies * number_of_directions * 4,
        time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        byte_order="<",
        float_size=4,
    )
    resource = _FakeResource(meta_data.record_size_bytes)
    return RestartFile(grid=grid, meta_data=meta_data, resource=resource)


def test_interpolate_in_space_excludes_masked_corner_from_depth():
    # Regression test for #14: _get_depth initialized its output buffer
    # with zeros and never excluded masked/land corners, so they were
    # silently averaged into the bilinear depth interpolation as depth=0
    # instead of being excluded like the sibling _get_data getter does.
    restart_file = _make_two_cell_restart_file()

    spectrum = restart_file.interpolate_in_space(
        latitude=numpy.array([0.5, 1.5]), longitude=numpy.array([0.5, 0.5])
    )

    # Cell A (all valid corners) and cell B (one masked corner) are both
    # 100m everywhere they're not masked; the masked corner in cell B must
    # not pull its weighted average down towards 0.
    assert numpy.all(numpy.abs(spectrum.depth.values - 100.0) < 1e-6)


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

    assert isinstance(spectra, Spectrum)
    assert spectra.number_of_spectra == 2

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

    assert isinstance(spectra, Spectrum)
    assert spectra.number_of_spectra == 3

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38966825, 3.31241975, 3.26012738])
    assert numpy.all(numpy.abs(spectra.hm0() - hm0) < 1e-3)

    # Fancy indices
    spectra = restart_file[[20000, 20001, 20002]]
    lats, lons = restart_file.coordinates([20000, 20001, 20002])

    assert isinstance(spectra, Spectrum)
    assert spectra.number_of_spectra == 3

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38966825, 3.31241975, 3.26012738])
    assert numpy.all(numpy.abs(spectra.hm0() - hm0) < 1e-3)

    # Scalar index
    spectra = restart_file[20000]
    lats, lons = restart_file.coordinates(20000)

    assert isinstance(spectra, Spectrum)
    assert spectra.number_of_spectra == 1

    # Check if coordinates are returned correctly
    assert numpy.all(numpy.abs(lats - spectra.latitude) < 1e-3)
    assert numpy.all(numpy.abs(lons - spectra.longitude) < 1e-3)

    # Check if we get the correct significant waveheights.
    hm0 = numpy.array([3.38966825])
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
