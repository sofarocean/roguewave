"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Module defining io operations performed on restart files.

Public Classes:
- N/A

Functions:

- `clone_restart_file`, helper function to clone an existing restart file.
    Mostly useful for testing purposes of our restart-file writing
    (otherwise we would use copy).
- `open_restart_file`, open a restart file and return a RestartFile object.
- `reassemble_restart_file_from_parts`, reassemble a complete restart file from
    parts. Part of a reduce operation after we conducted operations on part
    of the restart file in parallel (different instances).
- `write_partial_restart_file`, write a partial restart file as output. Part of
    a map operation, where we only operate on a small part of the restart file.
- `write_restart_file`, write a new restart file given updated spectra.

How To Use This Module
======================
(See the individual functions for details.)

1. import
2. open a restart file with open_restart_file
3. perform any of the read write operations using write_restart_file,
   write_partial_restart_file or reassemble_restart_file_from_parts
"""

import numpy
from roguewave.wavewatch3.model_definition import read_model_definition
from roguewave.wavewatch3.resources import create_resource
from roguewave.wavewatch3.restart_file import RestartFile
from roguewave.wavewatch3 import RestartFileTimeStack
from roguewave.wavewatch3.restart_file_metadata import read_header
from roguewave import FrequencyDirectionSpectrum
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from typing import Union
from xarray import Dataset, DataArray


def open_restart_file(restart_file: str, model_definition_file: str) -> RestartFile:
    """
    Open a restart file locally or remote and return a restart file object.

    :param restart_file: path or uri to a valid restart file
    :param model_definition_file: path or uri to model definition file that
        corresponds to the restart file.
    :return: A restart file object
    """
    restart_file_resource = create_resource(restart_file)
    model_definition_resource = create_resource(model_definition_file)

    meta_data = read_header(restart_file_resource)
    grid = read_model_definition(model_definition_resource)
    return RestartFile(grid, meta_data, restart_file_resource)


def open_restart_file_stack(
    uris, model_definition_file, cache=False, cache_name=None
) -> RestartFileTimeStack:
    """
    Open a restart file locally or remote and return a restart file object.

    :param restart_file: path or uri to a valid restart file
    :param model_definition_file: path or uri to model definition file that
        corresponds to the restart file.
    :return: A restart file object
    """
    restart_files = []
    model_definition_resource = create_resource(
        model_definition_file, "rb", cache, cache_name
    )
    grid = read_model_definition(model_definition_resource)

    for uri in uris:
        restart_file_resource = create_resource(uri, "rb", cache, cache_name)
        meta_data = read_header(restart_file_resource)
        restart_files.append(RestartFile(grid, meta_data, restart_file_resource))

    return RestartFileTimeStack(restart_files)


def reassemble_restart_file_from_parts(target_file, locations, source_restart_file):
    """
    Reassemble a valid restart file from partial restart files and write the
    result to the target file. This is used as part of a reduce operation to
    create a valid restart file from partial files created in batches in a map
    operation.

    :param target_file: Location to construct new restart file.
    :param locations:  Locations (remote or local) to read partial restart
        files from.
    :param model_definition_file: Location (remote or local) of model
        definition file.
    :return: None
    """

    # Open the partial restart files
    data = []

    def _worker(location):
        partial_spectra_reader = _PartialRestartFileReader(
            location, source_restart_file
        )
        return partial_spectra_reader.start, partial_spectra_reader.spectra()

    with ThreadPool(processes=10) as pool:
        data = list(tqdm(pool.imap(_worker, locations), total=len(locations)))

    # note that file is a Resource Object here.
    with create_resource(target_file, "wb") as resource:
        # Write the header
        resource.write(source_restart_file.header_bytes())

        # Write the partial spectra sorted by start index of the spectra
        for i_start, partial_spectra in tqdm(
            sorted(data, key=lambda x: x[0]), total=len(data)
        ):
            # Write to file
            resource.write(partial_spectra.tobytes("C"))

        # write the tail
        resource.write(source_restart_file.tail_bytes())


def write_restart_file(
    spectra: Union[Dataset, numpy.ndarray, DataArray, FrequencyDirectionSpectrum],
    target_file: str,
    parent_restart_file: RestartFile,
    spectra_are_frequence_energy_density=True,
) -> None:
    """
    Create a resource file from the given spectra at the target location. The
    target can be a local file or a s3 uri.

    :param spectra: Energy density as funciton of frequency or Action density
        as function of frequency. Needs to have the same number of spectra as
        the parent restart file.
    :param target_file: path to local file or s3 uri to be created.
    :param parent_restart_file: Restart file object of the parent restart file.
        We need this information to create valid header information.
    :param spectra_are_frequence_energy_density: are the spectra energy
        densities. If so we need to transform into wavenumber action density.
        If not, the spectra are assumed to already be valid wavenumber spectra
        and no transformation is applied.
    :return: None
    """
    if isinstance(spectra, (Dataset, FrequencyDirectionSpectrum)):
        spectra = spectra.variance_density.values
    elif isinstance(spectra, DataArray):
        spectra = spectra.values

    dtype = numpy.dtype("float32")  #
    dtype = dtype.newbyteorder(parent_restart_file._meta_data.byte_order)

    shape = spectra.shape
    if shape[0] != parent_restart_file.number_of_spatial_points:
        raise ValueError(
            "Input spectra have more spatial points than the "
            "source resource file contains"
        )

    if shape[1] != parent_restart_file.number_of_frequencies:
        raise ValueError(
            "Input spectra have more frequences than the "
            "source resource file contains"
        )

    if shape[2] != parent_restart_file.number_of_frequencies:
        raise ValueError(
            "Input spectra have more directions than the "
            "source resource file contains"
        )

    # Ensure we are writing the right floating point accuracy and byte_order
    spectra = spectra.astype(dtype, copy=False)
    if spectra_are_frequence_energy_density:
        conversion_factor = parent_restart_file.to_wavenumber_action_density(
            slice(None, None, None)
        )
        spectra[:, :, :] = spectra[:, :, :] * conversion_factor

    with create_resource(target_file, "wb") as file:
        # Use the parent file to get the valid header.
        file.write(parent_restart_file.header_bytes())
        file.write(spectra.tobytes("C"))
        # Use the parent file to get the "tail", i.e. all the information
        # stored in the restart file after the header. Currently *I* do not
        # know what this information exactly is.
        file.write(parent_restart_file.tail_bytes())


def write_partial_restart_file(
    spectra: Union[FrequencyDirectionSpectrum, Dataset, numpy.ndarray, DataArray],
    target_file: str,
    parent_restart_file: RestartFile,
    s: slice,
    spectra_are_frequence_energy_density=True,
) -> None:
    """
    Create a partial restart file containing only the given spectra and
    their corresponding linear indices in the complete restart file. Used as
    part of a map operation storing the information that was processed in the
    current batch while operating on the total spectra. See
    reassemble_restart_file_from_parts for the reduce operation to create a
    valid restart file from all its parts.

    :param spectra: Energy density as funciton of frequency or Action density
        as function of frequency. Needs to have the same number of spectra as
        the parent restart file.
    :param target_file: path to local file or s3 uri to be created. Needs to
        be unique for the partial files (w.r.t. the other partial files).
    :param parent_restart_file: Restart file object of the parent restart file.
        We need this information to create valid header information.
    :param s: slice- contains the start and end linear_indices. Note that only
        a stepsize of 1 is supported in a slice. And indices *must* be given
        (None is not valid).
    :param spectra_are_frequence_energy_density: are the spectra energy
        densities. If so we need to transform into wavenumber action density.
        If not, the spectra are assumed to already be valid wavenumber spectra
        and no transformation is applied.
    :return:
    """
    if isinstance(spectra, (Dataset, FrequencyDirectionSpectrum)):
        spectra = spectra.variance_density.values
    elif isinstance(spectra, DataArray):
        spectra = spectra.values

    dtype = numpy.dtype("float32")  #
    dtype = dtype.newbyteorder(parent_restart_file._meta_data.byte_order)

    shape = spectra.shape
    start, stop, step = s.indices(parent_restart_file.number_of_spatial_points)
    if shape[0] != stop - start:
        raise ValueError(
            "Input spectra have more spatial points than the "
            "source resource file contains"
        )

    if shape[1] != parent_restart_file.number_of_frequencies:
        raise ValueError(
            "Input spectra have more frequences than the "
            "source resource file contains"
        )

    if shape[2] != parent_restart_file.number_of_frequencies:
        raise ValueError(
            "Input spectra have more directions than the "
            "source resource file contains"
        )

    # Ensure we are writing the right floating point accuracy and byte_order
    spectra = spectra.astype(dtype, copy=False)

    if spectra_are_frequence_energy_density:
        spectra[:, :, :] = spectra[
            :, :, :
        ] * parent_restart_file.to_wavenumber_action_density(slice(start, stop, step))

    with create_resource(target_file, "wb") as file:
        location = parent_restart_file.resource.resource_location.encode("utf-8")

        # for a partial file we will just write as a header:
        #  1) the length of the location string (path, uri) of the
        #     source restart file
        #  2) the location string in utf-8
        #  3) the start index in int32, little endian.
        #  4) the stop index in int32, little endian.
        #  5) the spectra themselves.
        # All other information (spectral size etc.) is contained in the
        # source restart file we point to.
        file.write(len(location).to_bytes(4, "little"))
        file.write(location)
        file.write(start.to_bytes(4, "little"))
        file.write(stop.to_bytes(4, "little"))
        file.write(spectra.tobytes("C"))


class _PartialRestartFileReader:
    """
    Helper Class to read partial restart files. A partial restart file is a
    binary file that contains (in order):

        location_length (4 byte int): number of bytes needed for the char array
            that contains the parent file names.
        location (location_length bytes): char array contaning the location of
            the parent restart file (local file path or s3 uri).
        start (4 byte int): start linear index of the arrays
        stop (4 byte int): stop linear index (exclusive) of the arrays.
        spectra: ( start-stop)*numer_of_spectral_points*4 byte float array
            containing the spectral data.
    """

    def __init__(self, location, parent_restart_file):
        """
        :param location: local file path or s3 uri.
        :param parent_restart_file: restart file the partial spectra originate
            from. We need this to be able to write a valid restart file as
            output.
        """
        self.resource = create_resource(location)

        # Read the header information
        self.location_length = int.from_bytes(self.resource.read(4), "little")
        self.location = self.resource.read(self.location_length).decode("utf-8")

        self.start = int.from_bytes(self.resource.read(4), "little")
        self.stop = int.from_bytes(self.resource.read(4), "little")

        self.source_restart_file = parent_restart_file
        assert self.source_restart_file.resource.resource_location == self.location

    def spectra(self) -> numpy.ndarray:
        """
        Read the spectra contained in the partial spectral file and return
        a numpy.ndarray of type float32 with the correct byte-order.
        :return:
        """
        raw_data = self.resource.read()
        dtype = numpy.dtype("float32")
        dtype = dtype.newbyteorder(self.source_restart_file._meta_data.byte_order)
        spectra = numpy.frombuffer(raw_data, dtype=dtype)
        spectra = numpy.reshape(
            spectra,
            (
                self.stop - self.start,
                self.source_restart_file.number_of_frequencies,
                self.source_restart_file.number_of_directions,
            ),
        )
        return spectra


def clone_restart_file(source_restart_file, model_definition_file, target) -> None:
    """
    Clone restart file. Mostly useful to test that the reading and writing
    operations create an exact duplicate of a given restart file (otherwise
    we could just copy the file).

    :param source_restart_file: file to be cloned
    :param model_definition_file: model definition file
    :param target: target file
    :return: None
    """
    source_restart_file = open_restart_file(source_restart_file, model_definition_file)
    write_restart_file(source_restart_file[:], target, source_restart_file, True)
