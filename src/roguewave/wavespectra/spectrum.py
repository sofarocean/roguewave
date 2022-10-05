import numpy
from roguewave.interpolate.dataset import (
    interpolate_dataset_grid,
    interpolate_dataset_along_axis,
)
from roguewave.tools.math import wrapped_difference
from roguewave.tools.time import to_datetime64
from roguewave.wavetheory.lineardispersion import inverse_intrinsic_dispersion_relation
from roguewave.wavespectra.estimators import mem2
from typing import Iterator, Hashable, TypeVar, Union, List
from xarray import Dataset, DataArray, open_dataset
from xarray.core.coordinates import DatasetCoordinates

_NAME_F = "frequency"
_NAME_D = "direction"
_NAME_T = "time"
_NAME_E = "variance_density"
_NAME_a1 = "a1"
_NAME_b1 = "b1"
_NAME_a2 = "a2"
_NAME_b2 = "b2"
_NAME_LAT = "latitude"
_NAME_LON = "longitude"
_NAME_DEPTH = "depth"
_NAMES_2D = (_NAME_F, _NAME_D, _NAME_T, _NAME_E, _NAME_LAT, _NAME_LON, _NAME_DEPTH)
_NAMES_1D = (
    _NAME_F,
    _NAME_T,
    _NAME_E,
    _NAME_LAT,
    _NAME_LON,
    _NAME_a1,
    _NAME_b1,
    _NAME_a2,
    _NAME_b2,
    _NAME_DEPTH,
)
_SPECTRAL_VARS = (_NAME_E, _NAME_a1, _NAME_b1, _NAME_a2, _NAME_b2)
_SPECTRAL_DIMS = (_NAME_F, _NAME_D)
_SPACE_TIME_DIMS = (_NAME_T, _NAME_LON, _NAME_LAT)

_T = TypeVar("_T")


class DatasetWrapper:
    """
    A class that wraps a dataset object and passes through some of its primary
    functionality (get/set etc.). Used here mostly to make explicit what parts
    of the Dataset interface we actually expose in frequency objects. Note that
    we do not claim- or try to obtain completeness here. If full capabilities
    of the dataset object are needed we can simple operate directly on the
    dataset object itself.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def __setitem__(self, key, value) -> None:
        return self.dataset.__setitem__(key, value)

    def __copy__(self: _T) -> _T:
        cls = self.__class__
        return cls(self.dataset.copy())

    def __len__(self):
        return len(self.dataset)

    def copy(self: _T, deep=True) -> _T:
        if deep:
            return self.__deepcopy__({})
        else:
            return self.__copy__()

    def __deepcopy__(self: _T, memodict) -> _T:
        cls = self.__class__
        return cls(self.dataset.copy(deep=True))

    def coords(self) -> DatasetCoordinates:
        return self.dataset.coords

    def keys(self):
        return self.dataset.keys()

    def __contains__(self, key: object) -> bool:
        return key in self.dataset

    def __iter__(self) -> Iterator[Hashable]:
        return self.dataset.__iter__()


class WaveSpectrum(DatasetWrapper):
    frequency_units = "Hertz"
    angular_units = "Degrees"
    spectral_density_units = "m**2/Hertz"
    angular_convention = (
        "Wave travel direction (going-to), " "measured anti-clockwise from East"
    )
    bulk_properties = (
        "m0",
        "hm0",
        "tm01",
        "tm02",
        "peak_period",
        "peak_direction",
        "peak_directional_spread",
        "mean_direction",
        "mean_directional_spread",
        "peak_frequency",
        "peak_wavenumber",
        "latitude",
        "longitude",
        "time",
    )

    def __init__(self, dataset: Dataset):
        super(WaveSpectrum, self).__init__(dataset)

    def __add__(self: _T, other: _T) -> _T:
        spectrum = self.copy(deep=True)
        spectrum.dataset[_NAME_E] = spectrum.dataset[_NAME_E] + other.dataset[_NAME_E]
        return spectrum

    def __sub__(self: _T, other: _T) -> _T:
        spectrum = self.copy(deep=True)
        spectrum.dataset[_NAME_E] = spectrum.dataset[_NAME_E] - other.dataset[_NAME_E]
        return spectrum

    def __neg__(self: _T) -> _T:
        """
        Negate self- that is -spectrum
        :return: spectrum with all spectral values taken to have the opposite
            sign.
        """
        spectrum = self.copy(deep=True)
        spectrum.dataset[_NAME_E] = -spectrum.dataset[_NAME_E]
        return spectrum

    def __len__(self):
        return self.number_of_spectra

    def __getitem__(self: _T, item) -> _T:
        if len(item) < self.ndims:
            raise ValueError(
                f"Indexing requires same number of inputs as dimensions: {self.ndims}"
            )

        nspectral_dims = len(self.dims_spectral)
        space_time_index = item[:-nspectral_dims]

        dataset = Dataset()
        for var in self.dataset:
            if var in _SPECTRAL_VARS:
                dataset = dataset.assign({var: self.dataset[var].__getitem__(item)})
            else:
                if space_time_index:
                    # array
                    dataset = dataset.assign(
                        {var: self.dataset[var].__getitem__(space_time_index)}
                    )
                else:
                    # Scalar
                    dataset = dataset.assign({var: self.dataset[var]})

        cls = type(self)
        return cls(dataset)

    @property
    def ndims(self):
        return len(self.dims)

    def sel(self: _T, *args, **kwargs) -> _T:
        cls = type(self)
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].sel(*args, **kwargs)})
        return cls(dataset=dataset)

    def isel(self: _T, *args, **kwargs) -> _T:
        cls = type(self)
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].isel(*args, **kwargs)})
        return cls(dataset=dataset)

    def mean(self: _T, dim, skipna=False) -> _T:
        """
        Calculate the mean value of the spectrum along the given dimension.
        :param dim: dimension to average over
        :param skipna: whether or not to "skip" nan values; if True behaves as numpy.nanmean
        :return:
        """
        if dim in _SPECTRAL_DIMS:
            raise ValueError("Cannot calculate mean over spectral dimensions")

        cls = type(self)
        dataset = Dataset()
        # Todo: fix averaging over longitude for (prime/anti) meridian issues
        dataset = dataset.assign({dim: self.dataset[dim].mean(dim=dim, skipna=skipna)})
        for x in self.dataset:
            dataset = dataset.assign({x: self.dataset[x].mean(dim=dim, skipna=skipna)})
        return cls(dataset)

    def sum(self: _T, dim, skipna=False) -> _T:
        """
        Calculate the sum value of the spectrum along the given dimension.
        :param dim: dimension to sum over
        :param skipna: whether or not to "skip" nan values; if True behaves as numpy.nansum
        :return:
        """

        if dim in _SPECTRAL_DIMS:
            raise ValueError("Cannot calculate sum over spectral dimensions")

        cls = type(self)
        dataset = Dataset()
        # we assign the average coordinate to the dimension we sum over
        dataset = dataset.assign({dim: self.dataset[dim].mean(dim=dim, skipna=skipna)})
        for x in self.dataset:
            dataset = dataset.assign({x: self.dataset[x].sum(dim=dim, skipna=skipna)})
        return cls(dataset)

    def std(self: _T, dim, skipna=False) -> _T:
        """
        Calculate the standard deviation of the spectrum along the given dimension.
        :param dim: dimension to calculate standard deviation over
        :param skipna: whether or not to "skip" nan values; if True behaves as numpy.nanstd
        :return:
        """
        if dim in _SPECTRAL_DIMS:
            raise ValueError(
                "Cannot calculate standard deviation over spectral dimensions"
            )

        cls = type(self)
        dataset = Dataset()
        # we assign the average coordinate to the dimension we calculate the std over
        dataset = dataset.assign({dim: self.dataset[dim].mean(dim=dim, skipna=skipna)})
        for x in self.dataset:
            dataset = dataset.assign({x: self.dataset[x].std(dim=dim, skipna=skipna)})
        return cls(dataset)

    def shape(self):
        return self.variance_density.shape

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Calculate a "frequency moment" over the given range. A frequency moment
        here refers to the integral:

                    Integral-over-frequency-range[ e(f) * f**power ]

        :param power: power of the frequency
        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: frequency moment
        """
        _range = self._range(fmin, fmax)

        # Integrate dataset over frequencies. Make sure to fill any NaN entries with 0 before the integration.
        return (
            (self.e.isel({_NAME_F: _range}) * self.frequency[_range] ** power)
            .fillna(0)
            .integrate(coord=_NAME_F)
        )

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)

    @property
    def dims_space_time(self) -> List[str]:
        return [str(x) for x in self.variance_density.dims if x not in (_SPECTRAL_DIMS)]

    @property
    def dims_spectral(self) -> List[str]:
        return [str(x) for x in self.variance_density.dims if x in (_SPECTRAL_DIMS)]

    @property
    def dims(self) -> List[str]:
        return [str(x) for x in self.variance_density.dims]

    @property
    def number_of_spectra(self):
        dims = self.dims_space_time
        if dims:
            shape = 1
            for d in dims:
                shape *= len(self.dataset[d])
        else:
            return 1

    @property
    def spectral_values(self) -> DataArray:
        """
        :return: Spectral levels
        """
        return self.dataset[_NAME_E]

    @property
    def radian_frequency(self) -> DataArray:
        """
        :return: Radian frequency
        """
        data_array = self.dataset[_NAME_F] * 2 * numpy.pi
        data_array.name = "radian_frequency"
        return data_array

    @property
    def latitude(self) -> DataArray:
        """
        :return: latitudes
        """
        return self.dataset[_NAME_LAT]

    @property
    def longitude(self) -> DataArray:
        """
        :return: longitudes
        """
        return self.dataset[_NAME_LON]

    @property
    def time(self) -> DataArray:
        """
        :return: Time
        """
        return self.dataset[_NAME_T]

    @property
    def variance_density(self) -> DataArray:
        """
        :return: Time
        """
        return self.dataset[_NAME_E]

    @property
    def e(self) -> DataArray:
        """
        :return: 1D spectral values (directionally integrated spectrum).
            Equivalent to self.spectral_values if this is a 1D spectrum.
        """
        return self.dataset[_NAME_E]

    @property
    def a1(self) -> DataArray:
        """
        :return: normalized Fourier moment cos(theta)
        """
        return self.dataset[_NAME_a1]

    @property
    def b1(self) -> DataArray:
        """
        :return: normalized Fourier moment sin(theta)
        """
        return self.dataset[_NAME_b1]

    @property
    def a2(self) -> DataArray:
        """
        :return: normalized Fourier moment cos(2*theta)
        """
        return self.dataset[_NAME_a2]

    @property
    def b2(self) -> DataArray:
        """
        :return: normalized Fourier moment sin(2*theta)
        """
        return self.dataset[_NAME_b2]

    @property
    def A1(self) -> DataArray:
        """
        :return: Fourier moment cos(theta)
        """
        return self.a1 * self.e

    @property
    def B1(self) -> DataArray:
        """
        :return: Fourier moment sin(theta)
        """
        return self.b1 * self.e

    @property
    def A2(self) -> DataArray:
        """
        :return: Fourier moment cos(2*theta)
        """
        return self.a2 * self.e

    @property
    def B2(self) -> DataArray:
        """
        :return: Fourier moment sin(2*theta)
        """
        return self.b2 * self.e

    @property
    def frequency(self) -> DataArray:
        """
        :return: Frequencies (Hz)
        """
        return self.dataset[_NAME_F]

    def m0(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Zero order frequency moment. Also referred to as variance or energy.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: variance/energy
        """
        return self.frequency_moment(0, fmin, fmax)

    def m1(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        First order frequency moment. Primarily used in calculating a mean
        period measure (Tm01)

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: first order frequency moment.
        """
        return self.frequency_moment(1, fmin, fmax)

    def m2(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Second order frequency moment. Primarily used in calculating the zero
        crossing period (Tm02)

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: Second order frequency moment.
        """
        return self.frequency_moment(2, fmin, fmax)

    def hm0(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Significant wave height estimated from the spectrum, i.e. waveheight
        h estimated from variance m0. Common notation in literature.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: Significant wave height
        """
        return 4 * numpy.sqrt(self.m0(fmin, fmax))

    def tm01(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Mean period, estimated as the inverse of the center of mass of the
        spectral curve under the 1d spectrum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: Mean period
        """
        return self.m0(fmin, fmax) / self.m1(fmin, fmax)

    def tm02(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Zero crossing period based on Rice's spectral estimate.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: Zero crossing period
        """
        return numpy.sqrt(self.m0(fmin, fmax) / self.m2(fmin, fmax))

    def peak_index(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Index of the peak frequency of the 1d spectrum within the given range
        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak indices
        """
        return self.e.where(self._range(fmin, fmax), 0).argmax(dim=_NAME_F)

    def peak_frequency(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Peak frequency of the spectrum, i.e. frequency at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak frequency
        """
        return self.dataset[_NAME_F][self.peak_index(fmin, fmax)]

    def peak_period(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Peak period of the spectrum, i.e. period at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak period
        """
        return 1 / self.peak_frequency(fmin, fmax)

    def peak_direction(self, fmin=0, fmax=numpy.inf) -> DataArray:
        index = self.peak_index(fmin, fmax)
        return self._mean_direction(
            self.a1.isel(**{_NAME_F: index}), self.b1.isel(**{_NAME_F: index})
        )

    def peak_directional_spread(self, fmin=0, fmax=numpy.inf) -> DataArray:
        index = self.peak_index(fmin, fmax)
        a1 = self.a1.isel(**{_NAME_F: index})
        b1 = self.b1.isel(**{_NAME_F: index})
        return self._spread(a1, b1)

    @staticmethod
    def _mean_direction(a1: _T, b1: _T) -> _T:
        return numpy.arctan2(b1, a1) * 180 / numpy.pi

    @staticmethod
    def _spread(a1: _T, b1: _T) -> _T:
        return numpy.sqrt(2 - 2 * numpy.sqrt(a1**2 + b1**2)) * 180 / numpy.pi

    @property
    def mean_direction_per_frequency(self) -> DataArray:
        return self._mean_direction(self.a1, self.b1)

    @property
    def mean_spread_per_frequency(self) -> DataArray:
        return self._spread(self.a1, self.b1)

    def _spectral_weighted(self, property: DataArray, fmin=0, fmax=numpy.inf):
        range = {_NAME_F: self._range(fmin, fmax)}

        property = property.fillna(0)
        return numpy.trapz(
            property.isel(**range) * self.e.isel(**range), self.frequency[range]
        ) / self.m0(fmin, fmax)

    def mean_direction(self, fmin=0, fmax=numpy.inf):
        return self._mean_direction(self.mean_a1(fmin, fmax), self.mean_b1(fmin, fmax))

    def mean_directional_spread(self, fmin=0, fmax=numpy.inf):
        return self._spread(self.mean_a1(fmin, fmax), self.mean_b1(fmin, fmax))

    def mean_a1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a1, fmin, fmax)

    def mean_b1(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.b1, fmin, fmax)

    def mean_a2(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.a2, fmin, fmax)

    def mean_b2(self, fmin=0, fmax=numpy.inf):
        return self._spectral_weighted(self.b2, fmin, fmax)

    @property
    def depth(self) -> DataArray:
        return self.dataset[_NAME_DEPTH]

    @property
    def wavenumber(self) -> DataArray:
        """
        Determine the wavenumbers for the frequencies in the spectrum. Note that since the dispersion relation depends
        on depth the returned wavenumber array has the dimensions associated with the depth array by the frequency
        dimension.

        :return: wavenumbers
        """

        # For numba (used in the dispersion relation) we need raw numpy arrays of the correct dimension
        depth = self.depth.expand_dims(dim=_NAME_F, axis=-1).values
        radian_frequency = self.radian_frequency.expand_dims(dim=self.depth.dims).values

        # Broadcasting does not work inside the numba implementaiton, we explicitly need to construct arrays of the
        # correct input dimension.
        depth_shape = depth.shape
        radian_frequency_shape = radian_frequency.shape

        depth = depth * numpy.ones(radian_frequency_shape)
        radian_frequency = numpy.ones(depth_shape) * radian_frequency

        # Construct the output coordinates and dimension of the data array
        return_dimensions = (*self.dims_space_time, _NAME_F)
        coords = {}
        for dim in return_dimensions:
            coords[dim] = self.dataset[dim].values

        return DataArray(
            data=inverse_intrinsic_dispersion_relation(radian_frequency, depth),
            dims=return_dimensions,
            coords=coords,
        )

    @property
    def wavelength(self) -> DataArray:
        return 2 * numpy.pi / self.wavenumber

    @property
    def peak_wavenumber(self) -> DataArray:
        index = self.peak_index()
        # Construct the output coordinates and dimension of the data array
        coords = {}
        for dim in self.dims_space_time:
            coords[dim] = self.dataset[dim].values

        return DataArray(
            data=inverse_intrinsic_dispersion_relation(
                self.radian_frequency[index].values, self.depth.values
            ),
            dims=self.dims_space_time,
            coords=coords,
        )

    def bulk_variables(self) -> Dataset:
        dataset = Dataset()
        dataset["significant_waveheight"] = self.significant_waveheight
        dataset["mean_period"] = self.mean_period
        dataset["peak_period"] = self.peak_period()
        dataset["peak_direction"] = self.peak_direction()
        dataset["peak_directional_spread"] = self.peak_directional_spread()
        dataset["mean_direction"] = self.mean_direction()
        dataset["mean_directional_spread"] = self.mean_directional_spread()
        dataset["peak_frequency"] = self.peak_frequency()
        dataset["latitude"] = self.latitude
        dataset["longitude"] = self.longitude
        dataset["timestamp"] = self.time
        return dataset

    @property
    def significant_waveheight(self) -> DataArray:
        return self.hm0()

    @property
    def mean_period(self) -> DataArray:
        return self.tm01()

    @property
    def zero_crossing_period(self) -> DataArray:
        return self.tm02()

    def interpolate(self: _T, coordinates) -> _T:
        return self.__class__(interpolate_dataset_grid(coordinates, self.dataset))

    def interpolate_frequency(self: _T, new_frequencies) -> _T:
        return self.__class__(
            interpolate_dataset_along_axis(
                new_frequencies, self.dataset, coordinate_name="frequency"
            )
        )

    def _range(self, fmin=0.0, fmax=numpy.inf) -> numpy.ndarray:
        return (self.dataset[_NAME_F].values >= fmin) & (
            self.dataset[_NAME_F].values < fmax
        )

    def save_as_netcdf(self, path):
        self.dataset.to_netcdf(path)


class FrequencyDirectionSpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencyDirectionSpectrum, self).__init__(dataset)
        for name in _NAMES_2D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-2]))

    def direction_step(self) -> DataArray:
        difference = wrapped_difference(
            numpy.diff(self.direction.values, append=self.direction[0]), period=360
        )
        return DataArray(
            data=difference, coords={_NAME_D: self.direction.values}, dims=[_NAME_D]
        )

    @property
    def radian_direction(self) -> DataArray:
        data_array = self.dataset[_NAME_D] * numpy.pi / 180
        data_array.name = "radian_direction"
        return data_array

    def _directionally_integrate(self, data_array: DataArray) -> DataArray:
        return (data_array * self.direction_step()).sum(_NAME_D, skipna=True)

    @property
    def e(self) -> DataArray:
        return self._directionally_integrate(self.dataset[_NAME_E])

    @property
    def a1(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[_NAME_E] * numpy.cos(self.radian_direction)
            )
            / self.e
        )

    @property
    def b1(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[_NAME_E] * numpy.sin(self.radian_direction)
            )
            / self.e
        )

    @property
    def a2(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[_NAME_E] * numpy.cos(2 * self.radian_direction)
            )
            / self.e
        )

    @property
    def b2(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[_NAME_E] * numpy.sin(2 * self.radian_direction)
            )
            / self.e
        )

    @property
    def direction(self) -> DataArray:
        return self.dataset[_NAME_D]

    def spectrum_1d(self) -> "FrequencySpectrum":
        return create_1d_spectrum(
            self.frequency.values,
            self.e.values,
            self.time.values,
            self.latitude.values,
            self.longitude.values,
            self.a1.values,
            self.b1.values,
            self.a2.values,
            self.b2.values,
            depth=self.depth.values,
            dims=self.dims_space_time + [_NAME_F],
        )

    @property
    def number_of_directions(self) -> int:
        return len(self.direction)


class FrequencySpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencySpectrum, self).__init__(dataset)
        for name in _NAMES_1D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-1]))

    def as_frequency_direction_spectrum(
        self, number_of_directions, method="mem2"
    ) -> "FrequencyDirectionSpectrum":

        direction = numpy.linspace(0, 360, number_of_directions, endpoint=False)
        radian_direction = direction * numpy.pi / 180

        input_dims = [str(x) for x in self.spectral_values.dims]
        input_shape = self.spectral_values.shape
        output_shape = (*input_shape, number_of_directions)

        loop_elements = numpy.prod(input_shape[:-1])
        output_array = numpy.empty(
            (loop_elements, self.number_of_frequencies, number_of_directions)
        )

        for index in range(loop_elements):
            indices = numpy.unravel_index(index, input_shape[:-1])
            indexers = {dim: num for dim, num in zip(input_dims[:-1], indices)}
            output_array[index, :, :] = (
                mem2(
                    radian_direction,
                    self.a1.isel(indexers).values,
                    self.b1.isel(indexers).values,
                    self.a2.isel(indexers).values,
                    self.b2.isel(indexers).values,
                )
                * numpy.pi
                / 180
                * self.e.isel(indexers).values[:, None]
            )
        output_array = numpy.reshape(output_array, output_shape)

        return create_2d_spectrum(
            frequency=self.frequency.values,
            direction=direction,
            variance_density=output_array,
            time=self.time.values,
            latitude=self.latitude.values,
            longitude=self.longitude.values,
            depth=self.depth.values,
            dims=list(self.variance_density.dims) + [_NAME_D],
        )


def create_1d_spectrum(
    frequency: numpy.ndarray,
    variance_density: numpy.ndarray,
    time: Union[numpy.ndarray, float],
    latitude: Union[numpy.ndarray, float],
    longitude: Union[numpy.ndarray, float],
    a1: numpy.ndarray = None,
    b1: numpy.ndarray = None,
    a2: numpy.ndarray = None,
    b2: numpy.ndarray = None,
    depth: Union[numpy.ndarray, float] = numpy.inf,
    dims=(_NAME_T, _NAME_F),
) -> FrequencySpectrum:
    if a1 is None:
        a1 = numpy.nan + numpy.ones_like(variance_density)
    if b1 is None:
        b1 = numpy.nan + numpy.ones_like(variance_density)
    if a2 is None:
        a2 = numpy.nan + numpy.ones_like(variance_density)
    if b2 is None:
        b2 = numpy.nan + numpy.ones_like(variance_density)

    variables = {
        _NAME_T: numpy.atleast_1d(to_datetime64(time)),
        _NAME_LAT: numpy.atleast_1d(latitude),
        _NAME_LON: numpy.atleast_1d(longitude),
        _NAME_DEPTH: numpy.atleast_1d(depth),
        _NAME_F: frequency,
        _NAME_E: variance_density,
        _NAME_a1: a1,
        _NAME_b1: b1,
        _NAME_a2: a2,
        _NAME_b2: b2,
    }

    return FrequencySpectrum(create_spectrum_dataset(dims, variables))


def create_2d_spectrum(
    frequency: numpy.ndarray,
    direction: numpy.ndarray,
    variance_density: numpy.ndarray,
    time,
    latitude: Union[numpy.ndarray, float],
    longitude: Union[numpy.ndarray, float],
    dims=(_NAME_T, _NAME_F, _NAME_D),
    depth: Union[numpy.ndarray, float] = numpy.inf,
) -> FrequencyDirectionSpectrum:
    """
    :param frequency:
    :param direction:
    :param variance_density:
    :param time:
    :param latitude:
    :param longitude:
    :param dims:
    :param depth:
    :return:
    """

    variables = {
        _NAME_T: numpy.atleast_1d(to_datetime64(time)),
        _NAME_LAT: numpy.atleast_1d(latitude),
        _NAME_LON: numpy.atleast_1d(longitude),
        _NAME_DEPTH: numpy.atleast_1d(depth),
        _NAME_F: frequency,
        _NAME_D: direction,
        _NAME_E: variance_density,
    }
    return FrequencyDirectionSpectrum(create_spectrum_dataset(dims, variables))


def create_spectrum_dataset(dims, variables) -> Dataset:
    independent_variables = dims
    dependent_variables = [x for x in variables if x not in independent_variables]

    spectral_coords = {k: variables[k] for k in independent_variables}
    spatial_coords = {
        k: variables[k] for k in independent_variables if k not in _SPECTRAL_DIMS
    }

    dataset = Dataset()
    for variable in dependent_variables:
        if variable in _SPECTRAL_VARS:
            coords = spectral_coords
        else:
            coords = spatial_coords
        dims = [k for k in coords]

        if dims:
            dataset = dataset.assign(
                {
                    variable: DataArray(
                        data=variables[variable], dims=dims, coords=coords
                    )
                }
            )

        else:
            if len(variables[variable]) == 1:
                # If no coordinate is known, and the variable has length 1, we add it as a scalar.
                data = variables[variable][0]
            else:
                # otherwise we add without coordinate/dimension.
                data = DataArray(data=variables[variable])

            dataset = dataset.assign({variable: data})
    return dataset


def load_spectrum_from_netcdf(filename_or_obj) -> WaveSpectrum:
    """
    Load a spectrum from netcdf file
    :param filename_or_obj:
    :return:
    """
    dataset = open_dataset(filename_or_obj=filename_or_obj)
    if _NAME_D in dataset.coords:
        return FrequencyDirectionSpectrum(dataset=dataset)
    else:
        return FrequencySpectrum(dataset=dataset)
