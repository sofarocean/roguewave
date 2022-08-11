import numpy

from roguewave.wavewatch3.io import Resource, create_resource
from roguewave.wavewatch3.restart_file_metadata import read_header
from roguewave.wavewatch3.model_definition import Grid, read_model_definition
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation, \
    jacobian_wavenumber_to_radial_frequency

# It just needs to be large _enough_
from roguewave.wavewatch3.restart_file_metadata import MetaData
from typing import Sequence, Union


class FrequencySpectrum():
    _start_record = 2
    _dtype = numpy.dtype("float32").newbyteorder("<")

    def __init__(self, grid:Grid, meta_data:MetaData, reader:Resource, depth=None):
        self._grid = grid
        self._meta_data = meta_data
        self._reader = reader

        if depth is None:
            self._depth = numpy.inf \
            * numpy.ones((self.number_of_spatial_points,),dtype='float32')
        else:
            self._depth = depth


    @property
    def frequency(self) -> numpy.ndarray:
        return self._grid.frequencies

    @property
    def direction(self) -> numpy.ndarray:
        return self._grid.directions
    
    @property
    def latitude(self) -> numpy.ndarray:
        return self._grid.latitude

    @property
    def longitude(self) -> numpy.ndarray:
        return self._grid.longitude

    @property
    def number_of_directions(self) -> int:
        return len(self.direction)

    @property
    def number_of_frequencies(self) -> int:
        return len(self.frequency)
    
    @property
    def number_of_latitudes(self) -> int:
        return len(self.longitude)

    @property
    def number_of_longitudes(self) -> int:
        return len(self.longitude)

    @property
    def number_of_spatial_points(self) -> int:
        return self._grid.number_of_spatial_points

    @property
    def number_of_frequency_points(self) -> int:
        return self.number_of_frequencies * self.number_of_directions

    def __len__(self):
        return self.number_of_frequencies

    def __getitem__(self, s):
        if isinstance(s, Sequence) or isinstance(s, numpy.ndarray):
            data = self._fancy_index(s)
        else:
            data = self._sliced_index(s)

        jacobian = calculate_jacobian_factor( self._depth[s],
                                              self.frequency )
        return data * jacobian[:,:,None]

    def _sliced_index(self,s:slice):
        if not isinstance(s, slice):
            s = slice(s, s + 1, 1)

        start,stop,step = s.indices(self.number_of_spatial_points)

        if step != 1:
            raise ValueError('Cannot slice with stepsize different from 1')

        byte_slice = slice( self._byte_index(start),
                            self._byte_index(stop),step
                            )

        # Read raw data, cast as numpy array,
        data = self._reader.read_range(byte_slice)[0]
        data = numpy.frombuffer(data, dtype=self._dtype)
        return numpy.reshape(
            data , (
                s.stop - s.start,
                self.number_of_frequencies,
                self.number_of_directions
            )
        )


    def _fancy_index(self, indices:Union[Sequence,numpy.ndarray]):
        if isinstance(indices, Sequence):
            indices = numpy.array(indices,dtype='int32')

        slices = [ slice(self._byte_index(index),self._byte_index(index+1),1)
                   for index in indices ]
        data_for_slices = self._reader.read_range(slices)
        out = []
        for data in data_for_slices:
            data = numpy.frombuffer(data, dtype=self._dtype)
            out.append(
                numpy.reshape(
                    data , (
                        self.number_of_frequencies,
                        self.number_of_directions
                    )
                )
            )
        return numpy.array(out)


    def _byte_index(self,index):
        offset = self._meta_data.record_size_bytes * self._start_record
        return offset + index * self._meta_data.record_size_bytes


def calculate_jacobian_factor(
    depth: numpy.array, frequencies: numpy.array
) -> numpy.array:

    """
    # Jacobian is:
    # 1) transformation from k-> omega     (jacobian_wavenumber_to_radial_frequency() )
    # 2) transformation from omega -> f    ( 2 * pi )
    # 3) transformation from radians -> degrees   ( pi / 180.)
    """


    # depth = numpy.reshape( depth, ( len(depth),1 ))
    # w     = numpy.reshape( w    , (len(depth), 1))


    depth = depth[:, None]
    w = frequencies[None, :] * numpy.pi * 2
    w = numpy.repeat(w, len(depth), axis=0)

    wavenumbers = inverse_intrinsic_dispersion_relation(w, depth)

    jacobian = (
        jacobian_wavenumber_to_radial_frequency(wavenumbers, depth)
        * numpy.pi**2
        / 90.0
    )

    # Data is Action Density so also multiply with w
    return jacobian * w

def create_spectra( restart_file , model_definition_file  ) -> FrequencySpectrum:
    restart_file_resource = create_resource(restart_file)
    model_definition_resource = create_resource(model_definition_file)

    meta_data = read_header(restart_file_resource)
    grid, depth, mask = read_model_definition(model_definition_resource)
    return FrequencySpectrum(grid, meta_data, restart_file_resource,depth)