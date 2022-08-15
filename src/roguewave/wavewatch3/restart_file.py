import numpy

from roguewave.wavewatch3.resources import Resource
from roguewave.wavewatch3.model_definition import Grid, \
    LinearIndexedGridData
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation, \
    jacobian_wavenumber_to_radial_frequency
from roguewave.wavewatch3.restart_file_metadata import MetaData
from typing import Sequence, Union
from roguewave.interpolate.points import interpolate_points_nd
from datetime import datetime
from functools import cache


class RestartFile():
    _start_record = 2

    def __init__(self, grid:Grid, meta_data:MetaData,
                 resource:Resource, depth:LinearIndexedGridData=None,
                 convert_to_freq_energy_dens=True):
        self._grid = grid
        self._meta_data = meta_data
        self.resource = resource
        self._dtype = numpy.dtype("float32").newbyteorder(meta_data.byte_order)
        self._convert = convert_to_freq_energy_dens

        if depth is None:
            _depth = numpy.inf \
                * numpy.ones((self.number_of_spatial_points,),dtype='float32')
            self._depth = LinearIndexedGridData(_depth,grid)
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
    def number_of_spectral_points(self) -> int:
        return self.number_of_frequencies * self.number_of_directions

    @property
    def time(self) -> datetime:
        return self._meta_data.time

    def __len__(self):
        return self.number_of_frequencies

    def __getitem__(self, s):
        if isinstance(s, Sequence) or isinstance(s, numpy.ndarray):
            data = self._fancy_index(s)
        else:
            data = self._sliced_index(s)

        if self._convert:
            jacobian = self.to_frequency_energy_density(s)
            return data * jacobian[:,:,None]
        else:
            return data

    def _sliced_index(self,s:slice):
        if isinstance(s, int):
            s = slice(s, s + 1, 1)

        elif not isinstance( s, slice ):
            raise ValueError(f'Cannot use type {type(s)} as a spatial index.'
                             f' Use a slice or int instead.')

        start,stop,step = s.indices(self.number_of_spatial_points)
        start = start + self._start_record
        stop = stop + self._start_record
        if step != 1:
            # We cannot use contiguous IO in this case, recast as a fancy index
            indices = numpy.arange( start, stop, step=step )
            return self._fancy_index(indices)

        byte_slice = slice( self._byte_index(start),
                            self._byte_index(stop),1
                            )

        # Read raw data, cast as numpy array,
        number_of_spatial_points = stop-start
        if number_of_spatial_points == self.number_of_spatial_points:
            # If we read all data we just call the read function. This makes
            # little difference for a local file- but for an aws object this
            # allows us to use the much more efficient get_object.
            self.resource.seek( self._byte_index(self._start_record) )
            data = self.resource.read()
        else:
            data = self.resource.read_range(byte_slice)[0]
        data = numpy.frombuffer(data, dtype=self._dtype,
            count=self.number_of_spectral_points*number_of_spatial_points)

        return numpy.reshape(
            data , (
                stop - start,
                self.number_of_frequencies,
                self.number_of_directions
            )
        )


    def _fancy_index(self, indices:Union[Sequence,numpy.ndarray]):
        if isinstance(indices, Sequence):
            indices = numpy.array(indices,dtype='int32')

        indices = indices + self._start_record
        slices = [ slice(self._byte_index(index),self._byte_index(index+1),1)
                   for index in indices ]

        data_for_slices = self.resource.read_range(slices)
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
        return index * self._meta_data.record_size_bytes

    def interpolate(self, latitude, longitude):

        points = {"latitude":numpy.atleast_1d(latitude),
                  "longitude":numpy.atleast_1d(longitude)}

        coordinates = [("latitude",self.latitude),
                       ("longitude",self.longitude)]

        periodic_coordinates = {"longitude":360}

        def _get_data(  indices ):
            # To note- latitudes are the fast index, so this is swapped from
            # the "lat,lon" in the fortran code (leading index is fast index
            # in Fortran) to "lon,lat" here.
            index = self._grid.to_linear_index[indices[1],indices[0] ]

            output = numpy.zeros( (len(index),
                                     self.number_of_frequencies,
                                     self.number_of_directions ) )
            mask = index >= 0
            output[ mask,:,:] = self.__getitem__(index[mask])
            output[~mask,:,:] =numpy.nan
            return output

        output_shape = ( len(points['latitude']),
                         self.number_of_frequencies,
                         self.number_of_directions)

        return interpolate_points_nd(
            coordinates, points, periodic_coordinates,_get_data,
            period_data=None,discont=360, output_shape=output_shape
        )

    def to_wavenumber_action_density(
            self, s:Union[slice, numpy.ndarray, Sequence] ) -> numpy.array:
        return 1 / self.to_frequency_energy_density(s)

    def to_frequency_energy_density(
            self, s:Union[slice, numpy.ndarray, Sequence]
                                    ) -> numpy.array:

        """
        To convert a wavenumber Action density as a function of radial
        frequency (as stored by ww3) to a frequency Energy density we need to
        multiply the action density with the angular frequency (Action -> to
        Energy) and the proper Jacobians of the transformations. Specficically
                 #
        # 1) transformation from k-> omega
            (jacobian_wavenumber_to_radial_frequency() )
        # 2) transformation from omega -> f    ( 2 * pi )
        # 3) transformation from radians -> degrees   ( pi / 180.)
        """

        # get the depth and make sure it has dimension [ spatial_points , 1 ]
        depth = self._depth[s, None]

        # Get omega and make sure it has dimensions [ 1 , frequency_points ]
        w = self.frequency[None, :] * numpy.pi * 2

        # Repeat w along the spatial axis
        w = numpy.repeat(w, len(depth), axis=0)

        # get the wavenumber
        k = inverse_intrinsic_dispersion_relation(w, depth)

        # Calculate the various (Jacobian) factors
        jac_k_to_w = jacobian_wavenumber_to_radial_frequency(k, depth)
        jac_omega_f = 2*numpy.pi
        jac_rad_to_deg = numpy.pi/180
        action_to_energy = w

        # Return the result
        return action_to_energy * jac_rad_to_deg * jac_omega_f * jac_k_to_w

    @cache
    def header_bytes(self) -> bytes:
        s = slice(self._byte_index(0),
                  self._byte_index(self._start_record),1)
        return self.resource.read_range(s)[0]

    @cache
    def tail_bytes(self) -> bytes:
        self.resource.seek(self._byte_index(self._start_record
                                            + self.number_of_spatial_points))
        return self.resource.read()

    def number_of_header_bytes(self) -> int:
        a = self.header_bytes()
        return len(self.header_bytes())

    def number_of_tail_bytes(self) -> int:
        return len(self.tail_bytes())

    def size_in_bytes(self):
        return self.number_of_tail_bytes() + self.number_of_header_bytes() + \
               self.number_of_spatial_points * \
               self.number_of_spectral_points * self._dtype.itemsize

