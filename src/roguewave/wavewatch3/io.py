import numpy

from roguewave.wavewatch3.model_definition import read_model_definition
from roguewave.wavewatch3.resources import create_resource
from roguewave.wavewatch3.restart_file import RestartFile
from roguewave.wavewatch3.restart_file_metadata import read_header
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def open_restart_file( restart_file, model_definition_file) -> RestartFile:
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
    grid, depth, mask = read_model_definition(model_definition_resource)
    return RestartFile(grid, meta_data, restart_file_resource, depth)


def reassemble_restart_file_from_parts( target_file,
                                        locations,
                                        source_restart_file):
    """
    Reassemble a valid restart file from partial restart files and write the
    result to the target file.

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
        partial_spectra_reader = PartialRestartFileReader( location,
                                                           source_restart_file )
        return partial_spectra_reader.start, partial_spectra_reader.spectra()

    with ThreadPool(processes=10) as pool:
        data = list(
            tqdm(
                pool.imap(_worker, locations),
                total=len(locations)
            )
        )


    with create_resource(target_file,'wb') as file:
        # Write the header
        file.write(source_restart_file.header_bytes())

        # Write the partial spectra sorted by start index of the spectra
        for i_start, partial_spectra in tqdm(
                sorted(data, key=lambda x: x[0]),total=len(data)):

            # Write to file
            file.write(partial_spectra.tobytes('C'))

        # write the tail
        file.write(source_restart_file.tail_bytes())


def write_restart_file( spectra:numpy.ndarray,
                        target_file,
                        restart_file:RestartFile,
                        spectra_are_frequence_energy_density=True
                        ):
    """

    :param spectra:
    :param target_file:
    :param restart_file:
    :param s:
    :return:
    """

    dtype = numpy.dtype('float32') #
    dtype = dtype.newbyteorder( restart_file._meta_data.byte_order )

    shape = spectra.shape
    if shape[0] != restart_file.number_of_spatial_points:
        raise ValueError('Input spectra have more spatial points than the '
                         'source resource file contains')

    if shape[1] != restart_file.number_of_frequencies:
        raise ValueError('Input spectra have more frequences than the '
                         'source resource file contains')

    if shape[2] != restart_file.number_of_frequencies:
        raise ValueError('Input spectra have more directions than the '
                         'source resource file contains')

    # Ensure we are writing the right floating point accuracy and byte_order
    spectra = spectra.astype(dtype, copy=False)
    if spectra_are_frequence_energy_density:
        conversion_factor = \
            restart_file.to_wavenumber_action_density( slice( None,None,None ) )
        spectra[:,:,:] = spectra[:,:,:] * conversion_factor[ :,:,None]


    with create_resource(target_file,'wb') as file:
        file.write( restart_file.header_bytes() )
        file.write( spectra.tobytes('C') )
        file.write( restart_file.tail_bytes())


def write_partial_restart_file(
        spectra: numpy.ndarray,
        target_file: str,
        restart_file: RestartFile,
        s: slice,
        spectra_are_frequence_energy_density=True
        ):
    """

    :param spectra:
    :param target_file:
    :param restart_file:
    :param s:
    :return:
    """

    dtype = numpy.dtype('float32')  #
    dtype = dtype.newbyteorder(restart_file._meta_data.byte_order)

    shape = spectra.shape
    start, stop, step = s.indices(restart_file.number_of_spatial_points)
    if shape[0] != stop - start:
        raise ValueError('Input spectra have more spatial points than the '
                         'source resource file contains')

    if shape[1] != restart_file.number_of_frequencies:
        raise ValueError('Input spectra have more frequences than the '
                         'source resource file contains')

    if shape[2] != restart_file.number_of_frequencies:
        raise ValueError('Input spectra have more directions than the '
                         'source resource file contains')

    # Ensure we are writing the right floating point accuracy and byte_order
    spectra = spectra.astype(dtype, copy=False)

    if spectra_are_frequence_energy_density:
        spectra[:, :, :] = spectra[:, :, :] * \
                           restart_file.to_wavenumber_action_density(
                               slice(start, stop, step))[:, :, None]

    with create_resource(target_file,'wb') as file:
        location = restart_file.resource.resource_location.encode('utf-8')

        # for a partial file we will just write as a header:
        #  1) the length of the location string (path, uri) of the
        #     source restart file
        #  2) the location string in utf-8
        #  3) the start index in int32, little endian.
        #  4) the stop index in int32, little endian.
        #  5) the spectra themselves.
        # All other information (spectral size etc.) is contained in the
        # source restart file we point to.
        file.write(len(location).to_bytes(4, 'little'))
        file.write(location)
        file.write(start.to_bytes(4, 'little'))
        file.write(stop.to_bytes(4, 'little'))
        file.write(spectra.tobytes('C'))


class PartialRestartFileReader:
    def __init__(self, location, source_restart_file):

        self.resource = create_resource(location)

        # Read the header information
        self.location_length = int.from_bytes(self.resource.read(4), 'little')
        self.location = self.resource.read(
            self.location_length).decode('utf-8')

        self.start = int.from_bytes(self.resource.read(4), 'little')
        self.stop = int.from_bytes(self.resource.read(4), 'little')

        self.source_restart_file = source_restart_file
        assert self.source_restart_file.resource.resource_location \
                   == self.location

    def spectra(self):
        raw_data = self.resource.read()
        dtype = numpy.dtype('float32')
        dtype = dtype.newbyteorder(
            self.source_restart_file._meta_data.byte_order)
        spectra = numpy.frombuffer( raw_data, dtype=dtype )
        spectra = numpy.reshape( spectra,
            (
            self.stop-self.start ,
            self.source_restart_file.number_of_frequencies,
            self.source_restart_file.number_of_directions)
        )
        return spectra


def clone_restart_file( restart_file,
                        model_definition_file, target) -> None:

    restart_file = open_restart_file( restart_file ,
                                      model_definition_file )
    restart_file._convert = False
    write_restart_file( restart_file[:], target, restart_file,False )
