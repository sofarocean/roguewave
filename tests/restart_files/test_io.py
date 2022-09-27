from roguewave.wavewatch3.io import (
    write_partial_restart_file,
    reassemble_restart_file_from_parts,
)
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import os

from tests.restart_files import (
    REMOTE_HASH,
    REASSEMBLED_HASH,
    TEST_DIR,
    LOCAL_FILE_NAME,
    REMOTE_WRITE_PATH,
    file_hash,
    clone_remote,
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


def test_remote_partial_write():
    restart_file = clone_remote()
    number_of_chunks = 100
    chunksize = restart_file.number_of_spatial_points // number_of_chunks
    if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
        number_of_chunks += 1

    def _worker(arg):
        i_start, i_end, name = arg
        write_partial_restart_file(
            restart_file[i_start:i_end],
            name,
            restart_file,
            slice(i_start, i_end, 1),
            True,
        )

    arg = []
    for ii in range(0, number_of_chunks):
        i_start = ii * chunksize
        i_end = min((ii + 1) * chunksize, restart_file.number_of_spatial_points)
        name = f"{REMOTE_WRITE_PATH}chunk{ii:04d}"
        arg.append((i_start, i_end, name))

    with ThreadPool(processes=10) as pool:
        _ = list(tqdm(pool.imap(_worker, arg), total=len(arg)))


def test_remote_reassemble():
    restart_file = clone_remote()
    names = []
    number_of_chunks = 100
    chunksize = restart_file.number_of_spatial_points // number_of_chunks
    if number_of_chunks * chunksize < restart_file.number_of_spatial_points:
        number_of_chunks += 1

    for ii in range(0, number_of_chunks):
        names.append(f"{REMOTE_WRITE_PATH}chunk{ii:04d}")

    output = "test.file"
    reassemble_restart_file_from_parts(output, names, restart_file)
    assert file_hash(output) == REASSEMBLED_HASH
    os.remove(output)


if __name__ == "__main__":
    # test_clone_remote()
    # test_local_partial_write()
    # test_local_reassemble()
    test_remote_partial_write()
    test_remote_reassemble()
    #
