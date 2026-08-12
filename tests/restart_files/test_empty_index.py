from datetime import datetime, timezone
import numpy
from roguewave.wavewatch3.grid_tools import Grid
from roguewave.wavewatch3.restart_file import RestartFile
from roguewave.wavewatch3.restart_file_metadata import MetaData


def _make_minimal_restart_file():
    """
    A fully synthetic, in-memory RestartFile: Grid and MetaData are plain
    dataclasses with no file I/O, and the empty-index path under test never
    touches `resource`, so this needs no real restart file on disk or S3.
    """
    number_of_frequencies = 2
    number_of_directions = 4
    frequencies = numpy.array([0.1, 0.2])
    directions = numpy.array([0.0, 90.0, 180.0, 270.0])
    latitude = numpy.array([0.0, 1.0])
    longitude = numpy.array([0.0, 1.0])

    # A 2x2 lat/lon grid with a single sea point at (latitude[0], longitude[0]).
    to_linear_index = numpy.array([[0, -1], [-1, -1]])
    to_point_index = numpy.array([[0], [0]])  # [ilon, ilat] for linear index 0

    grid = Grid(
        number_of_spatial_points=1,
        frequencies=frequencies,
        directions=directions,
        latitude=latitude,
        longitude=longitude,
        depth=numpy.array([100.0]),
        mask=numpy.array([[1, 0], [0, 0]]),
        _to_linear_index=to_linear_index,
        _to_point_index=to_point_index,
    )
    meta_data = MetaData(
        name="test",
        version="1",
        grid_name="test",
        restart_type="test",
        nsea=1,
        nspec=number_of_frequencies * number_of_directions,
        record_size_bytes=number_of_frequencies * number_of_directions * 4,
        time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        byte_order="<",
        float_size=4,
    )
    return RestartFile(grid=grid, meta_data=meta_data, resource=None)


def test_getitem_with_empty_fancy_index_does_not_crash():
    restart_file = _make_minimal_restart_file()

    spectra = restart_file[numpy.array([], dtype="int32")]

    assert spectra.number_of_spectra == 0
    values = spectra.directional_variance_density.values
    assert values.shape == (0, 2, 4)


def test_fancy_index_with_empty_indices_has_correct_shape():
    restart_file = _make_minimal_restart_file()

    result = restart_file._fancy_index(numpy.array([], dtype="int32"))

    assert result.shape == (0, 2, 4)
