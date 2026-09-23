from datetime import datetime, timezone

import numpy
import pytest

from roguewave.wavewatch3.grid_tools import Grid
from roguewave.wavewatch3.io import (
    write_partial_restart_file,
    write_restart_file,
    reassemble_restart_file_from_parts,
    clone_restart_file,
    open_restart_file,
)
from roguewave.wavewatch3.restart_file import RestartFile
from roguewave.wavewatch3.restart_file_metadata import MetaData
import os
import shutil

from tests.restart_files import (
    REMOTE_HASH,
    REASSEMBLED_HASH,
    TEST_DIR,
    LOCAL_FILE_NAME,
    file_hash,
    clone_remote,
)


def _make_minimal_restart_file(number_of_frequencies, number_of_directions):
    """
    A fully synthetic, in-memory RestartFile with an unequal number of
    frequencies and directions, so shape-validation bugs that only show up
    when the two counts differ (e.g. #12) can be exercised without a real
    restart file on disk or S3 (the checked-in fixture used elsewhere in
    this file has number_of_frequencies == number_of_directions == 36,
    which cannot distinguish the two axes).
    """
    frequencies = numpy.linspace(0.1, 0.2, number_of_frequencies)
    directions = numpy.linspace(0.0, 360.0, number_of_directions, endpoint=False)
    latitude = numpy.array([0.0, 1.0])
    longitude = numpy.array([0.0, 1.0])

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


def test_write_restart_file_validates_against_direction_count():
    # Regression test for #12: write_restart_file validated the directions
    # axis (shape[2]) against number_of_frequencies instead of
    # number_of_directions, so a validly-shaped spectra array on a grid
    # where the two counts differ raised a spurious ValueError.
    restart_file = _make_minimal_restart_file(
        number_of_frequencies=2, number_of_directions=4
    )
    spectra = numpy.zeros((1, 2, 4), dtype="float32")
    output = "unused_write_restart_file_target.file"

    try:
        with pytest.raises(AttributeError):
            # A real write can't complete against this fixture's
            # resource=None -- reaching that failure (rather than the
            # shape-validation ValueError) confirms the valid shape passed
            # validation.
            write_restart_file(
                spectra,
                output,
                restart_file,
                spectra_are_frequence_energy_density=False,
            )
    finally:
        if os.path.exists(output):
            os.remove(output)

    # A genuinely wrong direction count is still correctly rejected.
    with pytest.raises(ValueError, match="directions"):
        write_restart_file(
            numpy.zeros((1, 2, 3), dtype="float32"),
            output,
            restart_file,
            spectra_are_frequence_energy_density=False,
        )


def test_write_partial_restart_file_validates_against_direction_count():
    # Regression test for #12, write_partial_restart_file's copy of the
    # same bug.
    restart_file = _make_minimal_restart_file(
        number_of_frequencies=2, number_of_directions=4
    )
    spectra = numpy.zeros((1, 2, 4), dtype="float32")
    output = "unused_write_partial_restart_file_target.file"

    try:
        with pytest.raises(AttributeError):
            write_partial_restart_file(
                spectra,
                output,
                restart_file,
                slice(0, 1, 1),
                spectra_are_frequence_energy_density=False,
            )
    finally:
        if os.path.exists(output):
            os.remove(output)

    with pytest.raises(ValueError, match="directions"):
        write_partial_restart_file(
            numpy.zeros((1, 2, 3), dtype="float32"),
            output,
            restart_file,
            slice(0, 1, 1),
            spectra_are_frequence_energy_density=False,
        )


def test_clone_remote():
    #
    local_file = os.path.join(TEST_DIR, LOCAL_FILE_NAME)
    clone_remote()
    assert file_hash(local_file) == REMOTE_HASH


def test_local_partial_write():
    restart_file = clone_remote()
    number_of_chunks = 100
    chunksize = restart_file.number_of_spatial_points // number_of_chunks
    if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
        number_of_chunks += 1

    for ii in range(0, number_of_chunks):
        i_start = ii * chunksize
        i_end = min((ii + 1) * chunksize, restart_file.number_of_spatial_points)
        name = f"chunk{ii:04d}"
        write_partial_restart_file(
            restart_file[i_start:i_end],
            name,
            restart_file,
            slice(i_start, i_end, 1),
            True,
        )


def test_clone_restart_file():
    # Regression test for #15: write_restart_file used to raise IndexError
    # when given a Spectrum/Dataset (e.g. via clone_restart_file's use of
    # RestartFile.__getitem__), because it read .variance_density instead of
    # .directional_variance_density.
    restart_file = clone_remote()
    local_file = os.path.join(TEST_DIR, LOCAL_FILE_NAME)
    model_definition_file = os.path.join(TEST_DIR, "mod_def.ww3")
    output = "cloned_restart_test.file"

    clone_restart_file(local_file, model_definition_file, output)

    try:
        cloned = open_restart_file(output, model_definition_file)
        assert cloned.number_of_spatial_points == restart_file.number_of_spatial_points
    finally:
        os.remove(output)


def test_local_reassemble():
    restart_file = clone_remote()
    names = []
    number_of_chunks = 100
    chunksize = restart_file.number_of_spatial_points // number_of_chunks
    if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
        number_of_chunks += 1

    for ii in range(0, number_of_chunks):
        names.append(f"chunk{ii:04d}")

    output = "test.file"
    reassemble_restart_file_from_parts(output, names, restart_file)
    assert file_hash(output) == REASSEMBLED_HASH

    os.remove(output)
    for filename in names:
        os.remove(filename)


def test_local_reassemble_in_place():
    # Regression test: target_file aliasing source_restart_file's own path
    # used to corrupt the output, because create_resource(target_file, "wb")
    # truncated the file before source_restart_file.header_bytes()/
    # .tail_bytes() were read back from it. Uses its own throwaway copy of
    # the fixture (not the shared cached restart001.ww3 that clone_remote()
    # reuses across tests), since this test overwrites the file it reads
    # from, and creates its own partial chunks rather than relying on
    # test_local_partial_write's leftover files (already removed by
    # test_local_reassemble by the time this runs).
    clone_remote()
    in_place_copy = os.path.join(TEST_DIR, "restart001_inplace_test.ww3")
    shutil.copyfile(os.path.join(TEST_DIR, LOCAL_FILE_NAME), in_place_copy)
    restart_file = open_restart_file(
        in_place_copy, os.path.join(TEST_DIR, "mod_def.ww3")
    )

    names = []
    number_of_chunks = 100
    chunksize = restart_file.number_of_spatial_points // number_of_chunks
    if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
        number_of_chunks += 1

    for ii in range(0, number_of_chunks):
        i_start = ii * chunksize
        i_end = min((ii + 1) * chunksize, restart_file.number_of_spatial_points)
        name = f"inplace_chunk{ii:04d}"
        write_partial_restart_file(
            restart_file[i_start:i_end],
            name,
            restart_file,
            slice(i_start, i_end, 1),
            True,
        )
        names.append(name)

    reassemble_restart_file_from_parts(in_place_copy, names, restart_file)
    assert file_hash(in_place_copy) == REASSEMBLED_HASH

    os.remove(in_place_copy)
    for filename in names:
        os.remove(filename)


# def test_remote_partial_write():
#     restart_file = clone_remote()
#     number_of_chunks = 100
#     chunksize = restart_file.number_of_spatial_points // number_of_chunks
#     if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
#         number_of_chunks += 1
#
#     def _worker(arg):
#         i_start, i_end, name = arg
#         write_partial_restart_file(
#             restart_file[i_start:i_end],
#             name,
#             restart_file,
#             slice(i_start, i_end, 1),
#             True,
#         )
#
#     arg = []
#     for ii in range(0, number_of_chunks):
#         i_start = ii * chunksize
#         i_end = min((ii + 1) * chunksize, restart_file.number_of_spatial_points)
#         name = f"{REMOTE_WRITE_PATH}chunk{ii:04d}"
#         arg.append((i_start, i_end, name))
#
#     with ThreadPool(processes=10) as pool:
#         _ = list(tqdm(pool.imap(_worker, arg), total=len(arg)))
#
#
# def test_remote_reassemble():
#     restart_file = clone_remote()
#     names = []
#     number_of_chunks = 100
#     chunksize = restart_file.number_of_spatial_points // number_of_chunks
#     if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
#         number_of_chunks += 1
#
#     for ii in range(0, number_of_chunks):
#         names.append(f"{REMOTE_WRITE_PATH}chunk{ii:04d}")
#
#     output = "test.file"
#     reassemble_restart_file_from_parts(output, names, restart_file)
#     assert file_hash(output) == REASSEMBLED_HASH
#     os.remove(output)


if __name__ == "__main__":
    test_clone_remote()
    test_local_partial_write()
    test_local_reassemble()
    # test_remote_partial_write()
    # test_remote_reassemble()
    #
