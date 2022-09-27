from dataclasses import dataclass
from typing import Union, Sequence, TypeVar

import numpy
from xarray import DataArray, Dataset

from roguewave.tools.math import wrapped_difference
from roguewave import FrequencyDirectionSpectrum
from roguewave.tools.time import to_datetime64


@dataclass()
class Grid:
    """
    A class that encapsulates all grid data from the model definition. It also
    contains convinience methods to project linear indexed data to a 2d array
    of latitude and longitude (with NaN for missing values), and vice versa
    extract the needed data to construct a linear indexed version compatible
    with WW3.

    Purpose:
    We need this information for reading restart files.
    """

    number_of_spatial_points: int  # number of spatial "sea" points
    frequencies: numpy.ndarray
    directions: numpy.ndarray
    latitude: numpy.ndarray  # 1D array of model latitudes
    longitude: numpy.ndarray  # 1D array of model longitudes
    depth: numpy.ndarray
    mask: numpy.ndarray
    _to_linear_index: numpy.ndarray  # mapping of [ilon,ilat] => linear index
    _to_point_index: numpy.ndarray  # mapping of linear index => [ilon,ilat]

    @property
    def _growth_factor(self):
        return self.frequencies[1] / self.frequencies[0]

    def longitude_index(
        self, linear_index: Union[slice, Sequence, numpy.ndarray]
    ) -> numpy.ndarray:
        """
        :param linear_index: linear 1d index
        :return: longitude 1d index
        """

        return self._to_point_index[0, linear_index]

    def latitude_index(
        self, linear_index: Union[slice, Sequence, numpy.ndarray]
    ) -> numpy.ndarray:
        """
        :param linear_index: linear 1d index
        :return: latitude 1d index
        """
        return self._to_point_index[1, linear_index]

    def index(
        self,
        latitude_index: Union[slice, Sequence, numpy.ndarray],
        longitude_index: Union[slice, Sequence, numpy.ndarray],
        valid_only=False,
    ):
        """
        get the linear index of the array
        :param latitude_index:
        :param longitude_index:
        :return:
        """
        indices = self._to_linear_index[latitude_index, longitude_index]
        if valid_only:
            indices = indices[indices >= 0]
        return indices

    def direction_step(self):
        return wrapped_difference(
            numpy.diff(self.directions, append=self.directions[0]), period=360
        )

    def frequency_step(self):
        delta = numpy.diff(
            self.frequencies,
            prepend=self.frequencies[0] / self._growth_factor,
            append=self.frequencies[-1] * self._growth_factor,
        )
        return (delta[0:-1] + delta[1:]) / 2

    def extract(
        self,
        s: Union[slice, Sequence, numpy.ndarray],
        var: Union[numpy.ndarray, DataArray],
    ) -> DataArray:
        """
        Extract linear indexed data (with indices indicated by the slice) from
        a 2D array (lat,lon)
        :param s: slice instance
        :param var: 2D ndarray
        :return: 1D ndarray of length slice.stop-slice.start
        """
        if isinstance(var, DataArray):
            return var[
                DataArray(self.latitude_index(s), dims="points"),
                DataArray(self.longitude_index(s), dims="points"),
            ]
        else:
            return DataArray(
                data=var[self.latitude_index(s), self.longitude_index(s)], dims="points"
            )

    def project(
        self,
        lon_slice: slice,
        lat_slice: slice,
        var: Union[numpy.ndarray, DataArray],
        except_val=numpy.nan,
    ) -> DataArray:
        """
        Project linear indexed data onto a 2d array (lat ,lon).

        :param lon_slice: slice of ilon indices we want
        :param lat_slice: slice of ilat indices we want
        :param var:
        :param except_val:
        :return:
        """
        ind = self._to_linear_index[lat_slice, lon_slice]
        mask = ind >= 0
        ind = ind[mask]
        ilon = self.longitude_index(ind)
        ilat = self.latitude_index(ind)

        nlon = len(self.longitude[lon_slice])
        nlat = len(self.latitude[lat_slice])

        # this will not work for stepsizes != 1
        out = numpy.zeros((nlat, nlon), dtype=var.dtype) + except_val
        out[ilat, ilon] = var[ind]

        return DataArray(
            data=out,
            coords={
                "latitude": self.latitude[lat_slice],
                "longitude": self.longitude[lon_slice],
            },
            dims=("latitude", "longitude"),
            name="variance",
        )

    def set_linear_data(
        self,
        lon_slice: Union[slice, Sequence, numpy.ndarray],
        lat_slice: Union[slice, Sequence, numpy.ndarray],
        linear_data: Union[numpy.ndarray, DataArray],
        data: Union[numpy.ndarray, DataArray],
    ):
        """
        Update data stored in a linear array with data from a 2d array.
        :param lon_slice: longitude indices
        :param lat_slice: latitude indices
        :param linear_data: linear data array to update
        :param data: 2d Array
        :return:
        """

        ind = self.index(lat_slice, lon_slice, valid_only=True)
        linear_data[ind] = self.extract(ind, data)
        return linear_data


def _unpack_ndarray(data: numpy.ndarray, grid: Grid, exception_value) -> numpy.ndarray:

    shape = data.shape
    if shape[0] != grid.number_of_spatial_points:
        raise ValueError(
            "First dimension of data needs to have the same number of points as the number of sea points"
            f"in the ww3 model. Given: {shape[0]}, expected: {grid.number_of_spatial_points}"
        )

    output = (
        numpy.zeros((len(grid.latitude), len(grid.longitude), *shape[1:]))
        + exception_value
    )
    ind = grid.index(slice(None, None, None), slice(None, None, None), True)

    ilon = grid.longitude_index(ind)
    ilat = grid.latitude_index(ind)
    output[ilat, ilon, ...] = data[:, ...]
    return output


def _unpack_data_array(data: DataArray, grid: Grid, exception_value) -> DataArray:
    dims = []
    output_lat_index = 0
    output_lon_index = 1
    for index, dim in enumerate(data.dims):
        if dim == "linear_index":
            dims += ["latitude", "longitude"]
            output_lat_index = index
            output_lon_index = index + 1
        else:
            dims += [dim]

    coords = {x: data[x] for x in data.dims if x != "linear_index"}
    coords["latitude"] = grid.latitude
    coords["longitude"] = grid.longitude

    output_shape = [len(coords[x]) for x in dims]
    output = numpy.zeros(output_shape) + exception_value

    ind = grid.index(slice(None, None, None), slice(None, None, None), True)
    ilon = grid.longitude_index(ind)
    ilat = grid.latitude_index(ind)

    output_indicer = [slice(None)] * len(output_shape)
    output_indicer[output_lat_index] = ilat
    output_indicer[output_lon_index] = ilon
    output[tuple(output_indicer)] = data.values
    return DataArray(data=output, dims=dims, coords=coords)


_T = TypeVar("_T", DataArray, numpy.ndarray, FrequencyDirectionSpectrum)


def unpack_ww3_data(data: _T, grid: Grid, exception_value=numpy.nan) -> _T:
    """
    Unpack wavewatch3 data stored on a linear grid. Data can scalars (e.g. waveheigths) or spectra. Formats can be numpy
    arrays, dataarrays or spectra object. The output replaces the linear indexed dimension with unpacked two dimensions
    (lat,lon). Locations where data is not given are set to the exception value. To note: if a numpy array is given,
    the linear indexed dimension must be the first dimension.

    :param data: data to unpack
    :param grid: ww3 grid description
    :return: unpacked data.
    """
    if isinstance(data, DataArray):
        return _unpack_data_array(data, grid, exception_value)

    elif isinstance(data, numpy.ndarray):
        return _unpack_ndarray(data, grid, exception_value)

    elif isinstance(data, FrequencyDirectionSpectrum):
        # For the spectral object we need to unpack the variance density, and the depth together.
        variance_density = _unpack_data_array(
            data.variance_density, grid, exception_value
        )
        depth = _unpack_data_array(data.depth, grid, exception_value)

        # Creating the output dataset. Note that "time" is not necessarily a dimension for spectra returned from the
        # wavewatch 3 restart file, but is a required entry. So we may need to add it manually.
        dataset = Dataset(
            data_vars={"variance_density": variance_density, "depth": depth}
        )
        if "time" not in dataset:
            dataset["time"] = to_datetime64(data.time.values)
        return FrequencyDirectionSpectrum(dataset)
