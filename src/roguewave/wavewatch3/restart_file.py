"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Module defining the restart file class. This is the primary class to interact
    with wavewatch3 restart file data.

Public Classes:
- `RestartFile`, file to interact with restart files from WW3.

Functions:
- N/A

How To Use This Module
======================
(See the individual functions for details.)

1. import
2. create a restart file object.
"""

import numpy
from roguewave.wavewatch3.resources import Resource
from roguewave.wavewatch3.model_definition import Grid, \
    LinearIndexedGridData
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation, \
    jacobian_wavenumber_to_radial_frequency
from roguewave.wavewatch3.restart_file_metadata import MetaData
from typing import Sequence, Union, Tuple, Mapping
from roguewave.interpolate.nd_interp import NdInterpolator
from datetime import datetime
from functools import cache
from roguewave.tools.time import to_datetime_utc, to_datetime64
from roguewave.interpolate.geometry import TrackSet
from xarray import DataArray, Dataset, concat


class RestartFile(Sequence):
    _start_record = 2

    def __init__(self,
                 grid:Grid,
                 meta_data:MetaData,
                 resource:Resource,
                 depth:LinearIndexedGridData=None,
                 return_freq_energy_density=True):

        self._grid = grid
        self._meta_data = meta_data
        self.resource = resource
        self._dtype = numpy.dtype("float32").newbyteorder(meta_data.byte_order)
        self._convert = return_freq_energy_density

        if depth is None:
            _depth = numpy.inf \
                * numpy.ones((self.number_of_spatial_points,),dtype='float32')
            self._depth = LinearIndexedGridData(_depth,grid)
        else:
            self._depth = depth

    def set_return_freq_energy_density(self):
        self._convert = True

    def set_return_k_action_density(self):
        self._convert = False

    @property
    def frequency(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of frequencies
        """
        return self._grid.frequencies

    @property
    def direction(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of directions
        """
        return self._grid.directions
    
    @property
    def latitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of latitudes.
        """
        return self._grid.latitude

    @property
    def longitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of longitudes.
        """
        return self._grid.longitude

    def coordinates(self, index:Union[slice,int,numpy.ndarray]
                    )->Tuple[Union[float,numpy.ndarray ],
                             Union[float,numpy.ndarray ]]:
        """
        Return the latitude and longitude as a function of the linear index.
        :param index: linear index
        :return:  ( latitude(s), longitude(s)
        """
        ilon = self._grid.longitude_index(index)
        ilat = self._grid.latitude_index(index)
        return self.latitude[ilat], self.longitude[ilon]

    @property
    def number_of_directions(self) -> int:
        """
        :return: number of directions
        """
        return len(self.direction)

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)
    
    @property
    def number_of_latitudes(self) -> int:
        """
        :return: number of latitudes.
        """
        return len(self.latitude)

    @property
    def number_of_longitudes(self) -> int:
        """
        :return: number of longitudes.
        """
        return len(self.longitude)

    @property
    def number_of_spatial_points(self) -> int:
        """
        :return: Number of spatial points in the restart file. This only counts
            the number of sea points and is *not* equal to
            self.number_of_latitudes * self.number_of_longitudes
            Also referred to as "NSEA" in wavewatch III.
        """
        return self._grid.number_of_spatial_points

    @property
    def number_of_spectral_points(self) -> int:
        """
        :return: number of spectral points.
        """
        return self.number_of_frequencies * self.number_of_directions

    @property
    def time(self) -> datetime:
        """
        :return: Valid time of the restart file.
        """
        return self._meta_data.time

    def __len__(self) -> int:
        """
        :return: We consider a restart file as a container of spectra, so its
            length is the number of spatial points.
        """
        return self.number_of_spatial_points

    def _sliced_index(self,s:slice) -> numpy.ndarray:
        """
        Get wavenumber action density spectra at sliced indices.
        :param s: slice
        :return: wavenumber spectra from sliced indices. Returned data is of
        type float32 and has the shape:
                (s.stop - s.start,
                number_of_frequencies,
                number_of_directions)
        """
        if isinstance(s, (int,numpy.int32,numpy.int64)):
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

    def __getitem__(self, s:Union[slice,numpy.ndarray,Sequence,int]
                    ) -> Dataset:
        """
        Dunder method for the Sequence protocol. If obj is an instance of
        RestartFile, this allows us to use `obj[10:13]` to get spectra with
        linear indices 10 to 12.

        Input can either be a integer, slice, or fancy indexing through
        a sequency of numbers or a 1d numpy integer array.

        :param s: slice, integer, 1d numpy integer array, Sequence of integers.
        :return: We return the requested spectra. Depending on how we setup
            the object we return frequency energy density spectra (default) or
            wavenumber spectra. Returned data is of type float32 and has
            the shape:
                (number_of_spatial_points_requested,
                number_of_frequencies,
                number_of_directions)
        """
        if isinstance(s, Sequence) or isinstance(s, numpy.ndarray):
            data = self._fancy_index(s)
        else:
            data = self._sliced_index(s)

        if self._convert:
            # Are we using raw wavenumbers or frequency spectra
            jacobian = self.to_frequency_energy_density(s)
            data = data * jacobian

        if isinstance(s,slice):
            s = list(range(*s.indices(self.number_of_spatial_points)))
        elif isinstance(s,(int,numpy.int32,numpy.int64)):
            s = [s]

        coords = self.coordinates(s)
        return Dataset(
            data_vars={
                "variance_density": (
                    ('point_index','frequency','direction'),data ),
                "longitude": (
                    ('point_index'),coords[1] ),
                "latitude": (
                    ('point_index'), coords[0]),
                "linear_index": (
                    ('point_index'), s),
            },
            coords={
                "point index": numpy.arange(0,len(s) ),
                'frequency':self.frequency,
                'direction':self.direction
            })


    def _fancy_index(self, indices:Union[Sequence,numpy.ndarray]
                     ) -> numpy.ndarray:
        """
        Get wavenumber action density spectra at given indices.
        :param indices: linear indices we want spectra from (Sequence or numpy
            array of integers).
        :return: wavenumber spectra from sliced indices. Returned data is of
        type float32 and has the shape:
                (len(indices),
                number_of_frequencies,
                number_of_directions)
        """
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

    def _byte_index(self,index) -> int:
        """
        Get a byte index of a requested record. Record sizes are always
        number_of_spectral_points * 4 bytes. Note that the first
        self._start_record's contain header information. Spectral data starts
        at index = self._start_record
        :param index: Linear index
        :return: Linear byte index
        """
        return index * self._meta_data.record_size_bytes

    def interpolate_in_space(self,
                             latitude:Union[numpy.ndarray,float],
                             longitude:Union[numpy.ndarray,float]) -> numpy.ndarray:
        """
        Extract interpolated spectra at given latitudes and longitudes.
        Input can be either a single latitude and longitude pair, or a
        numpy array of latitudes and longitudes.

        :param latitude: latitudes to get interpolated spectra
        :param longitude: longitudes to get interpolated spectra
        :return: Interpolated spectra. Returned data is of  type float32 and
        has the shape:
                (len(indices),
                number_of_frequencies,
                number_of_directions)
        """

        points = {"latitude":numpy.atleast_1d(latitude),
                  "longitude":numpy.atleast_1d(longitude)}

        coordinates = [("latitude",self.latitude),
                       ("longitude",self.longitude)]

        periodic_coordinates = {"longitude":360}

        def _get_data(  indices, _dummy ):
            index = self._grid.index(latitude_index=indices[0],
                                     longitude_index=indices[1])

            output = numpy.zeros( (len(index),
                                     self.number_of_frequencies,
                                     self.number_of_directions ) )
            mask = index >= 0
            output[ mask,:,:] = self.__getitem__(index[mask])
            output[~mask,:,:] =numpy.nan
            return output


        data_shape = [ len(self.latitude), len(self.longitude),
                       self.number_of_frequencies,self.number_of_directions]
        data_coordinates = (
            ('latitude', self.latitude),
            ('longitude', self.longitude),
            ('frequency',self.frequency),
            ('direction', self.direction)
        )

        interpolator = NdInterpolator(
            get_data=_get_data,
            data_coordinates=data_coordinates,
            data_shape=data_shape,
            interp_coord_names=list(points.keys()),
            interp_index_coord_name='time',
            data_periodic_coordinates=periodic_coordinates,
            data_period=None,
            data_discont=None
        )

        return interpolator.interpolate(points)

    def to_wavenumber_action_density(
            self, s:Union[slice, numpy.ndarray, Sequence] ) -> numpy.array:
        """
        Factor that when multiplied with frequency energy density spectra at
        the givem indices converts them to action wavenumber spectra. E.g:
            Ek = to_wavenumber_action_density(slice[10,20)) * Ef[10:20]

        :param s: slice for linear indices.
        :return: Multiplication factor as numpy ndarray of size.
                (number_of_elements_in_slice,number_of_frequencies,1)

        To note, we add the trailing 1 to make it easy to use broadcasting (
            the Jacobian is constant in directional space).
        """
        return 1 / self.to_frequency_energy_density(s)[:,:,None]

    def to_frequency_energy_density(
            self, s:Union[slice, numpy.ndarray, Sequence]
                                    ) -> numpy.array:

        """
        Factor that when multiplied with wavenumber action density spectra at
        the givem indices converts them to frequency energy spectra. E.g:
            Ef = to_frequency_energy_density(slice[10,20)) * Ek[10:20]

        :param s: slice for linear indices.
        :return: Multiplication factor as numpy ndarray of size.
                (number_of_elements_in_slice,number_of_frequencies,1)

        To note, we add the trailing 1 to make it easy to use broadcasting (
            the Jacobian is constant in directional space).

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
        factor = action_to_energy * jac_rad_to_deg * jac_omega_f * jac_k_to_w
        return factor[:,:,None]

    @cache
    def header_bytes(self) -> bytes:
        """
        Return the header bytes, the first self._record_start records. This is
        primarily used to created a new restart file that has the same metadata
        as the current restart file.

        We cache this information to allow for more rapid retrieval if the
        information is stored on s3.

        :return: raw bytes of the header.
        """
        s = slice(self._byte_index(0),
                  self._byte_index(self._start_record),1)
        return self.resource.read_range(s)[0]

    @cache
    def tail_bytes(self) -> bytes:
        """
        After all spectral data in the restart file there is a bunch of
        additional information regarding currents (needed for uniqueness of
        energy to action) and other information. We do not currently parse this
        information in any way. Instead, if we want to create an updated
        version of the restart file (e.g. for data assimilation) we merely
        append the information from the source (or background) file. This
        function is a helper method to retrieve that information.

        We cache this information to allow for more rapid retrieval if the
        information is stored on s3.

        :return: raw bytes of the tail.
        """
        self.resource.seek(self._byte_index(self._start_record
                                            + self.number_of_spatial_points))
        return self.resource.read()

    def variance(self, latitude_slice:slice, longitude_slice:slice
                 ) -> numpy.ndarray:
        """
        Calculate the variance at sliced indices for latitudes and longitudes.
        Returns a 2d array with constant latitudes along rows and
        constant longitudes along colunns

        :param latitude_slice: latitude index range as slice
        :param longitude_slice: longitude index range as slice
        :return:
        """
        index = self._grid.index(
            latitude_index=latitude_slice,
            longitude_index=longitude_slice,
            valid_only=True)

        linear_indexed_variance = self.variance_linear_index(index)
        return self._grid.project( lon_slice=longitude_slice,
                                   lat_slice=latitude_slice,
                                   var=linear_indexed_variance)

    def variance_linear_index(self, index) -> numpy.ndarray:
        """
        Calculate the variance at linear indices. Returns a 1d of variances
        at requested indices

        :param latitude_slice: latitude index range as slice
        :param longitude_slice: longitude index range as slice
        :return:
        """
        toggle = not self._convert
        if toggle:
            self.set_return_freq_energy_density()
        spectra = self[index]
        if toggle:
            self.set_return_k_action_density()

        delta_f = self._grid.frequency_step()[None,:,None]
        delta_dir = self._grid.direction_step()[None,None,:]
        return numpy.sum( delta_f*delta_dir*spectra,axis=(1,2) )


    def number_of_header_bytes(self) -> int:
        """
        :return: length of the header in bytes.
        """
        return len(self.header_bytes())

    def number_of_tail_bytes(self) -> int:
        """
        :return: length of the tail in bytes.
        """
        return len(self.tail_bytes())

    def size_in_bytes(self):
        """
        :return: Total size in bytes of a restart file.
        """
        return self.number_of_tail_bytes() + self.number_of_header_bytes() + \
               self.number_of_spatial_points * \
               self.number_of_spectral_points * self._dtype.itemsize


class RestartFileStack:
    def __init__(self, restart_files: Sequence[RestartFile]):
        self._restart_files = restart_files
        self._grid = restart_files[0]._grid
        self._time = to_datetime_utc([ x.time for x in restart_files ])

    @property
    def frequency(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of frequencies
        """
        return self._grid.frequencies

    @property
    def direction(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of directions
        """
        return self._grid.directions

    @property
    def latitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of latitudes.
        """
        return self._grid.latitude

    @property
    def longitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of longitudes.
        """
        return self._grid.longitude

    def coordinates(self, index: Union[slice, int, numpy.ndarray]
                    ) -> Tuple[Union[float, numpy.ndarray],
                               Union[float, numpy.ndarray]]:
        """
        Return the latitude and longitude as a function of the linear index.
        :param index: linear index
        :return:  ( latitude(s), longitude(s)
        """
        ilon = self._grid.longitude_index(index)
        ilat = self._grid.latitude_index(index)
        return self.latitude[ilat], self.longitude[ilon]

    @property
    def number_of_directions(self) -> int:
        """
        :return: number of directions
        """
        return len(self.direction)

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)

    @property
    def number_of_latitudes(self) -> int:
        """
        :return: number of latitudes.
        """
        return len(self.latitude)

    @property
    def number_of_longitudes(self) -> int:
        """
        :return: number of longitudes.
        """
        return len(self.longitude)

    @property
    def number_of_spatial_points(self) -> int:
        """
        :return: Number of spatial points in the restart file. This only counts
            the number of sea points and is *not* equal to
            self.number_of_latitudes * self.number_of_longitudes
            Also referred to as "NSEA" in wavewatch III.
        """
        return self._grid.number_of_spatial_points

    @property
    def number_of_spectral_points(self) -> int:
        """
        :return: number of spectral points.
        """
        return self.number_of_frequencies * self.number_of_directions

    @property
    def time(self) -> Sequence[datetime]:
        return self._time

    def __len__(self):
        return len(self._restart_files)

    def __getitem__(self, nargs):
        if len(nargs) == 2:
            time_index, linear_index = nargs
        elif len(nargs) == 3:
            time_index, lat_index, lon_index = nargs
            linear_index = self._grid.index(lat_index,
                                            lon_index,valid_only=True)
        else:
            raise ValueError('unexpected number of indices')

        if isinstance(time_index,(int,numpy.int32,numpy.int64)):
            time_index = [time_index]

        if isinstance(time_index,slice):
            time = self.time[time_index]
            time_index = list(range(*time_index.indices(len(self))))
            data = [self._restart_files[it][linear_index] for it in time_index]
        else:
            time = [self.time[it] for it in time_index]
            data = [self._restart_files[it][ilin] for it,ilin
                    in zip(time_index,linear_index)]

        data = concat( data , dim='time',  )
        data.coords['time'] = to_datetime64(time)
        return data


    def interpolate(self,latitude:Union[numpy.ndarray,float],
                        longitude:Union[numpy.ndarray,float],
                        time:Union[numpy.ndarray,numpy.datetime64,datetime]
                    ) -> Dataset:
        """
        Extract interpolated spectra at given latitudes and longitudes.
        Input can be either a single latitude and longitude pair, or a
        numpy array of latitudes and longitudes.

        :param latitude: latitudes to get interpolated spectra
        :param longitude: longitudes to get interpolated spectra
        :return: Interpolated spectra. Returned data is of  type float32 and
        has the shape:
                (len(indices),
                number_of_frequencies,
                number_of_directions)
        """
        time = to_datetime64(time)
        points = {"time":numpy.atleast_1d(time),
                  "latitude":numpy.atleast_1d(latitude),
                  "longitude":numpy.atleast_1d(longitude),
                  }

        coordinates = [("time",to_datetime64(self.time)),
                        ("latitude",self.latitude),
                       ("longitude",self.longitude),
                       ]

        periodic_coordinates = {"longitude":360}


        def _get_data(  indices , idims):
            time_index = indices[0]
            index = self._grid.index(latitude_index=indices[1],
                                     longitude_index=indices[2])

            output = numpy.zeros( (
                len(index),
                self.number_of_frequencies,
                self.number_of_directions ) )
            mask = index >= 0

            output[  mask,:,:] = numpy.squeeze(
                self.__getitem__((time_index[mask],index[mask]))['variance_density'])
            output[ ~mask,:,:] =numpy.nan
            return output

        data_shape = [ len(self.time), len(self.latitude), len(self.longitude),
                       self.number_of_frequencies,self.number_of_directions]
        data_coordinates = (
            ('time',to_datetime64(self.time)),
            ('latitude', self.latitude),
            ('longitude', self.longitude),
            ('frequency',self.frequency),
            ('direction', self.direction)
        )

        interpolator = NdInterpolator(
            get_data=_get_data,
            data_coordinates=data_coordinates,
            data_shape=data_shape,
            interp_coord_names=list(points.keys()),
            interp_index_coord_name='time',
            data_periodic_coordinates=periodic_coordinates,
            data_period=None,
            data_discont=None
        )
        dataset = interpolator.interpolate(points)
        return Dataset(
            data_vars={
                "variance_density": (
                    ('time','frequency','direction'),dataset ),
                "longitude": (
                    ('time'),longitude ),
                "latitude": (
                    ('time'),latitude),
            },
            coords={
                'time':time,
                'frequency':self.frequency,
                'direction':self.direction
            })

    def interpolate_tracks(self,tracks:TrackSet
                          ) -> Mapping[str,DataArray]:
        out = {}
        for _id,track in tracks.tracks.items():
            out[_id] = self.interpolate(track.latitude,
                                          track.longitude,track.time)
        return out
