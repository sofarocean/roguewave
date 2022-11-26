import numpy
from roguewave.interpolate.dataset import (
    interpolate_dataset_grid,
    interpolate_dataset_along_axis,
)
from roguewave.tools.math import wrapped_difference
from roguewave.tools.time import to_datetime64
from roguewave.wavetheory.lineardispersion import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_velocity,
)
from roguewave.wavespectra.estimators import (
    estimate_directional_distribution,
    Estimators,
)
from typing import Iterator, Hashable, TypeVar, Union, List, Literal, Mapping
from xarray import Dataset, DataArray, open_dataset, concat, ones_like, where
from xarray.core.coordinates import DatasetCoordinates
from warnings import warn

NAME_F: Literal["frequency"] = "frequency"
NAME_D: Literal["direction"] = "direction"
NAME_T: Literal["time"] = "time"
NAME_E: Literal["variance_density"] = "variance_density"
NAME_a1: Literal["a1"] = "a1"
NAME_b1: Literal["b1"] = "b1"
NAME_a2: Literal["a2"] = "a2"
NAME_b2: Literal["b2"] = "b2"
NAME_LAT: Literal["latitude"] = "latitude"
NAME_LON: Literal["longitude"] = "longitude"
NAME_DEPTH: Literal["depth"] = "depth"
NAMES_2D = (NAME_F, NAME_D, NAME_T, NAME_E, NAME_LAT, NAME_LON, NAME_DEPTH)
NAMES_1D = (
    NAME_F,
    NAME_T,
    NAME_E,
    NAME_LAT,
    NAME_LON,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
    NAME_DEPTH,
)
SPECTRAL_VARS = (NAME_E, NAME_a1, NAME_b1, NAME_a2, NAME_b2)
SPECTRAL_DIMS = (NAME_F, NAME_D)
SPACE_TIME_DIMS = (NAME_T, NAME_LON, NAME_LAT)

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
        spectrum.dataset[NAME_E] = spectrum.dataset[NAME_E] + other.dataset[NAME_E]
        return spectrum

    def __sub__(self: _T, other: _T) -> _T:
        spectrum = self.copy(deep=True)
        spectrum.dataset[NAME_E] = spectrum.dataset[NAME_E] - other.dataset[NAME_E]
        return spectrum

    def __neg__(self: _T) -> _T:
        """
        Negate self- that is -spectrum
        :return: spectrum with all spectral values taken to have the opposite
            sign.
        """
        spectrum = self.copy(deep=True)
        spectrum.dataset[NAME_E] = -spectrum.dataset[NAME_E]
        return spectrum

    def __len__(self):
        return self.number_of_spectra

    def __getitem__(self: _T, item) -> _T:

        if isinstance(item, tuple):
            if len(item) < self.ndims:
                raise ValueError(
                    f"Indexing requires same number of inputs as dimensions: {self.ndims}"
                )
            space_time_index = item[: -len(self.dims_spectral)]
        else:
            if not self.ndims == 1:
                raise ValueError(
                    f"Indexing requires same number of inputs as dimensions: {self.ndims}"
                )
            space_time_index = []

        dataset = Dataset()
        for var in self.dataset:
            if var in SPECTRAL_VARS:
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

    @property
    def frequency_step(self) -> DataArray:
        prepend = 2 * self.frequency[0] - self.frequency[1]
        append = 2 * self.frequency[-1] - self.frequency[-2]
        diff = numpy.diff(self.frequency, append=append, prepend=prepend)
        return DataArray(
            data=(diff[0:-1] * 0.5 + diff[1:] * 0.5),
            dims=NAME_F,
            coords={NAME_F: self.frequency},
        )

    def is_invalid(self) -> DataArray:
        return self.variance_density.isnull().all(dim=self.dims_spectral)

    def is_valid(self) -> DataArray:
        return ~self.is_invalid()

    def drop_invalid(self: _T) -> _T:
        return self._apply_filter(self.is_valid())

    def where(self: _T, condition) -> _T:
        return self._apply_filter(condition)

    def _apply_filter(self: _T, boolean_mask: DataArray) -> _T:
        dataset = Dataset()
        for var in self.dataset:
            data = self.dataset[var].where(
                boolean_mask.reindex_like(self.dataset[var]), drop=True
            )
            dataset = dataset.assign({var: data})

        cls = type(self)
        return cls(dataset)

    def mean(self: _T, dim, skipna=False) -> _T:
        """
        Calculate the mean value of the spectrum along the given dimension.
        :param dim: dimension to average over
        :param skipna: whether or not to "skip" nan values; if True behaves as numpy.nanmean
        :return:
        """
        if dim in SPECTRAL_DIMS:
            raise ValueError("Cannot calculate mean over spectral dimensions")

        cls = type(self)
        dataset = Dataset()
        # Todo: fix averaging over longitude for (prime/anti) meridian issues
        dataset = dataset.assign({dim: self.dataset[dim].mean(dim=dim, skipna=skipna)})
        for x in self.dataset:
            dataset = dataset.assign({x: self.dataset[x].mean(dim=dim, skipna=skipna)})
        return cls(dataset)

    def flatten(self: "WaveSpectrum", flattened_coordinate="linear_index") -> _T:
        """
        Serialize the non-spectral dimensions creating a single leading dimension without a coordinate.
        """

        # Get the current dimensions and shape
        dims = self.dims_space_time
        coords = self.coords_space_time
        shape = self.space_time_shape()
        length = numpy.product(shape)

        # Calculate the flattened shape
        new_shape = (length,)
        new_spectral_shape = (length, *self.spectral_shape())
        new_dims = [flattened_coordinate] + self.dims_spectral

        linear_index = DataArray(
            data=numpy.arange(0, length), dims=flattened_coordinate
        )
        indices = numpy.unravel_index(linear_index.values, shape)

        dataset = {}
        for index, dim in zip(indices, dims):
            dataset[dim] = DataArray(
                data=coords[dim].values[index], dims=flattened_coordinate
            )

        for name in self.dataset:
            if name in SPECTRAL_VARS:
                x = DataArray(
                    data=self.dataset[name].values.reshape(new_spectral_shape),
                    dims=new_dims,
                    coords=self.coords_spectral,
                )
            else:
                x = DataArray(
                    data=self.dataset[name].values.reshape(new_shape),
                    dims=flattened_coordinate,
                )
            dataset[name] = x

        cls = type(self)
        return cls(Dataset(dataset))

    def sum(self: _T, dim, skipna=False) -> _T:
        """
        Calculate the sum value of the spectrum along the given dimension.
        :param dim: dimension to sum over
        :param skipna: whether or not to "skip" nan values; if True behaves as numpy.nansum
        :return:
        """

        if dim in SPECTRAL_DIMS:
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
        if dim in SPECTRAL_DIMS:
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

    def spectral_shape(self):
        number_of_spectral_dims = len(self.dims_spectral)
        return self.shape()[-number_of_spectral_dims:]

    def space_time_shape(self):
        number_of_spectral_dims = len(self.dims_spectral)
        return self.shape()[:-number_of_spectral_dims]

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
            (self.e.isel({NAME_F: _range}) * self.frequency[_range] ** power)
            .fillna(0)
            .integrate(coord=NAME_F)
        )

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)

    @property
    def dims_space_time(self) -> List[str]:
        return [str(x) for x in self.variance_density.dims if x not in (SPECTRAL_DIMS)]

    @property
    def coords_space_time(self) -> Mapping[str, DataArray]:
        return {dim: self.dataset[dim] for dim in self.dims_space_time}

    @property
    def coords_spectral(self) -> Mapping[str, DataArray]:
        return {dim: self.dataset[dim] for dim in self.dims_spectral}

    @property
    def dims_spectral(self) -> List[str]:
        return [str(x) for x in self.variance_density.dims if x in (SPECTRAL_DIMS)]

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
            return shape
        else:
            return 1

    @property
    def spectral_values(self) -> DataArray:
        """
        :return: Spectral levels
        """
        return self.dataset[NAME_E]

    @property
    def radian_frequency(self) -> DataArray:
        """
        :return: Radian frequency
        """
        data_array = self.dataset[NAME_F] * 2 * numpy.pi
        data_array.name = "radian_frequency"
        return data_array

    @property
    def latitude(self) -> DataArray:
        """
        :return: latitudes
        """
        return self.dataset[NAME_LAT]

    @property
    def longitude(self) -> DataArray:
        """
        :return: longitudes
        """
        return self.dataset[NAME_LON]

    @property
    def time(self) -> DataArray:
        """
        :return: Time
        """
        return self.dataset[NAME_T]

    @property
    def variance_density(self) -> DataArray:
        """
        :return: Time
        """
        return self.dataset[NAME_E]

    @property
    def e(self) -> DataArray:
        """
        :return: 1D spectral values (directionally integrated spectrum).
            Equivalent to self.spectral_values if this is a 1D spectrum.
        """
        return self.dataset[NAME_E]

    @property
    def a1(self) -> DataArray:
        """
        :return: normalized Fourier moment cos(theta)
        """
        return self.dataset[NAME_a1]

    @property
    def b1(self) -> DataArray:
        """
        :return: normalized Fourier moment sin(theta)
        """
        return self.dataset[NAME_b1]

    @property
    def a2(self) -> DataArray:
        """
        :return: normalized Fourier moment cos(2*theta)
        """
        return self.dataset[NAME_a2]

    @property
    def b2(self) -> DataArray:
        """
        :return: normalized Fourier moment sin(2*theta)
        """
        return self.dataset[NAME_b2]

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
        return self.dataset[NAME_F]

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

    def wave_speed(self) -> DataArray:
        """
        :return:
        """

        # Note we multiply inverse wavenumber with frequency to force xarray to return a number_of_points by
        # by number of frequencies data structure.
        return (1 / self.wavenumber) * self.radian_frequency

    def peak_wave_speed(self) -> DataArray:
        return 2 * numpy.pi * self.peak_frequency() / self.peak_wavenumber

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
        return self.e.where(self._range(fmin, fmax), 0).argmax(dim=NAME_F)

    def peak_frequency(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Peak frequency of the spectrum, i.e. frequency at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak frequency
        """
        return self.dataset[NAME_F][self.peak_index(fmin, fmax)]

    def peak_angular_frequency(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Peak frequency of the spectrum, i.e. frequency at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak frequency
        """
        return self.peak_frequency(fmin, fmax) * numpy.pi * 2

    def peak_period(self, fmin=0, fmax=numpy.inf) -> DataArray:
        """
        Peak period of the spectrum, i.e. period at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :return: peak period
        """
        peak_period = 1 / self.peak_frequency(fmin, fmax)
        peak_period.name = "peak period"
        peak_period = peak_period.drop("frequency")
        return peak_period

    def peak_direction(self, fmin=0, fmax=numpy.inf) -> DataArray:
        index = self.peak_index(fmin, fmax)
        return self._mean_direction(
            self.a1.isel(**{NAME_F: index}), self.b1.isel(**{NAME_F: index})
        )

    def peak_directional_spread(self, fmin=0, fmax=numpy.inf) -> DataArray:
        index = self.peak_index(fmin, fmax)
        a1 = self.a1.isel(**{NAME_F: index})
        b1 = self.b1.isel(**{NAME_F: index})
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
        range = {NAME_F: self._range(fmin, fmax)}

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
        depth = self.dataset[NAME_DEPTH]
        return where(depth.isnull(), numpy.inf, depth)

    @property
    def group_velocity(self) -> DataArray:
        depth = self.depth.expand_dims(dim=NAME_F, axis=-1).values
        k = self.wavenumber.values
        depth = depth * numpy.ones(k.shape)

        # Construct the output coordinates and dimension of the data array
        return_dimensions = (*self.dims_space_time, NAME_F)
        coords = {}
        for dim in return_dimensions:
            coords[dim] = self.dataset[dim].values

        return DataArray(
            data=intrinsic_group_velocity(k, depth),
            dims=return_dimensions,
            coords=coords,
        )

    @property
    def wavenumber(self) -> DataArray:
        """
        Determine the wavenumbers for the frequencies in the spectrum. Note that since the dispersion relation depends
        on depth the returned wavenumber array has the dimensions associated with the depth array by the frequency
        dimension.

        :return: wavenumbers
        """

        # For numba (used in the dispersion relation) we need raw numpy arrays of the correct dimension
        depth = self.depth.expand_dims(dim=NAME_F, axis=-1).values
        radian_frequency = self.radian_frequency.expand_dims(dim=self.depth.dims).values

        # Broadcasting does not work inside the numba implementaiton, we explicitly need to construct arrays of the
        # correct input dimension.
        depth_shape = depth.shape
        radian_frequency_shape = radian_frequency.shape

        depth = depth * numpy.ones(radian_frequency_shape)
        radian_frequency = numpy.ones(depth_shape) * radian_frequency

        # Construct the output coordinates and dimension of the data array
        return_dimensions = (*self.dims_space_time, NAME_F)
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

    def extrapolate_tail(self, end_frequency, power=-5) -> "FrequencySpectrum":
        e = self.e
        a1 = self.a1
        b1 = self.b1
        a2 = self.a2
        b2 = self.b2

        frequency = self.frequency.values
        frequency_delta = frequency[-1] - frequency[-2]
        n = int((end_frequency - frequency[-1]) / frequency_delta) + 1

        fstart = frequency[-1] + frequency_delta
        fend = frequency[-1] + n * frequency_delta
        tail_frequency = numpy.linspace(fstart, fend, n, endpoint=True)
        tail_frequency = DataArray(
            data=tail_frequency, coords={"frequency": tail_frequency}, dims="frequency"
        )
        tail_power = e.isel(frequency=-1) * (tail_frequency / frequency[-1]) ** power
        tail_a1 = a1.isel(frequency=-1) * ones_like(tail_frequency)
        tail_b1 = b1.isel(frequency=-1) * ones_like(tail_frequency)
        tail_a2 = a2.isel(frequency=-1) * ones_like(tail_frequency)
        tail_b2 = b2.isel(frequency=-1) * ones_like(tail_frequency)

        e = concat((e, tail_power), dim="frequency")
        a1 = concat((a1, tail_a1), dim="frequency")
        b1 = concat((b1, tail_b1), dim="frequency")
        a2 = concat((a2, tail_a2), dim="frequency")
        b2 = concat((b2, tail_b2), dim="frequency")

        return create_1d_spectrum(
            e.frequency.values,
            e.values,
            e.time.values,
            self.latitude.values,
            self.longitude.values,
            a1.values,
            b1.values,
            a2.values,
            b2.values,
            depth=self.depth.values,
            dims=self.dims_space_time + [NAME_F],
        )

    def bandpass(self: _T, fmin=0, fmax=numpy.inf) -> _T:

        dataset = Dataset()

        for name in self.dataset:
            if name in SPECTRAL_VARS:
                data = self.dataset[name].where(
                    (self.frequency > fmin) & (self.frequency < fmax), drop=True
                )
                dataset = dataset.assign({name: data})
            else:
                dataset = dataset.assign({name: self.dataset[name]})
        cls = type(self)
        return cls(dataset)

    def interpolate_frequency(
        self: _T, new_frequencies, extrapolate_tail=True, extrapolate_power=5
    ) -> _T:
        return self.__class__(
            interpolate_dataset_along_axis(
                new_frequencies, self.dataset, coordinate_name="frequency"
            )
        )

    def _range(self, fmin=0.0, fmax=numpy.inf) -> numpy.ndarray:
        return (self.dataset[NAME_F].values >= fmin) & (
            self.dataset[NAME_F].values < fmax
        )

    def save_as_netcdf(self, path):
        self.dataset.to_netcdf(path)

    def multiply(
        self, array: numpy.ndarray, dimensions: List[str] = None, inplace=False
    ) -> _T:
        """
        Multiply the variance density with the given numpy array. Broadcasting is performed automatically if dimensions
        are provided. If no dimensions are provided the array needs to have the exact same shape as the variance
        density array.

        :param array: Array to multiply with variance density
        :param dimension: Dimensions of the array
        :return: self
        """
        if inplace:
            output = self
        else:
            output = self.copy()

        coords = {}
        shape = array.shape
        if dimensions is None:
            if shape != self.shape():
                raise ValueError(
                    "If no dimensions are provided the array must have the exact same shape as the"
                    "variance density array."
                )

            output.dataset[NAME_E] = self.dataset[NAME_E] * array
            return output

        if len(shape) != len(dimensions):
            raise ValueError(
                "The dimensions of the input array must match the number of dimension labels"
            )

        for length, dimension in zip(shape, dimensions):
            if dimension not in self.dims:
                raise ValueError(
                    f"Dimension {dimension} not a valid dimension of the spectral object."
                )
            coords[dimension] = self.dataset[dimension].values

            if len(self.dataset[dimension].values) != length:
                raise ValueError(
                    f"Array length along the dimension {dimension} does not match the length of the"
                    f"coordinate of the same name in the spctral object."
                )

        data = DataArray(data=array, coords=coords, dims=dimensions)
        output.dataset[NAME_E] = self.dataset[NAME_E] * data
        return output


class FrequencyDirectionSpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencyDirectionSpectrum, self).__init__(dataset)
        for name in NAMES_2D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-2]))

    @property
    def direction_step(self) -> DataArray:
        difference = wrapped_difference(
            numpy.diff(self.direction.values, append=self.direction[0]), period=360
        )
        return DataArray(
            data=difference, coords={NAME_D: self.direction.values}, dims=[NAME_D]
        )

    @property
    def radian_direction(self) -> DataArray:
        data_array = self.dataset[NAME_D] * numpy.pi / 180
        data_array.name = "radian_direction"
        return data_array

    def _directionally_integrate(self, data_array: DataArray) -> DataArray:
        return (data_array * self.direction_step).sum(NAME_D, skipna=True)

    @property
    def e(self) -> DataArray:
        return self._directionally_integrate(self.dataset[NAME_E])

    @property
    def a1(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[NAME_E] * numpy.cos(self.radian_direction)
            )
            / self.e
        )

    @property
    def b1(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[NAME_E] * numpy.sin(self.radian_direction)
            )
            / self.e
        )

    @property
    def a2(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[NAME_E] * numpy.cos(2 * self.radian_direction)
            )
            / self.e
        )

    @property
    def b2(self) -> DataArray:
        return (
            self._directionally_integrate(
                self.dataset[NAME_E] * numpy.sin(2 * self.radian_direction)
            )
            / self.e
        )

    @property
    def direction(self) -> DataArray:
        return self.dataset[NAME_D]

    def as_frequency_spectrum(self) -> "FrequencySpectrum":
        dataset = {
            "a1": self.a1,
            "b1": self.b1,
            "a2": self.a2,
            "b2": self.b2,
            "variance_density": self.e,
        }
        for name in self.dataset:
            if name not in SPECTRAL_VARS:
                dataset[name] = self.dataset[name]

        return FrequencySpectrum(Dataset(dataset))

    def spectrum_1d(self) -> "FrequencySpectrum":
        """
        Will be depricated
        :return:
        """
        warn(
            'spectrum_1d method will be removed, use "as_frequency_spectrum" instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.as_frequency_spectrum()

    @property
    def number_of_directions(self) -> int:
        return len(self.direction)


class FrequencySpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencySpectrum, self).__init__(dataset)
        for name in NAMES_1D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-1]))

    def interpolate(self: "FrequencySpectrum", coordinates) -> "FrequencySpectrum":
        """

        :param coordinates:
        :return:
        """
        _dataset = Dataset()
        _moments = [NAME_a1, NAME_b1, NAME_a2, NAME_b2]

        # For physical reasons it is better to interpolate the scaled moments - as opposed to the normalized moments.
        # For the dataset we interpolate we set a1: to A1 etc. Afterwards we scale the output back to the normalized
        # state.
        for name in self.dataset:
            _name = str(name)
            if _name in _moments:
                _dataset = _dataset.assign({_name: getattr(self, _name) * self.e})
            else:
                _dataset = _dataset.assign({_name: self.dataset[_name]})

        interpolated_data = interpolate_dataset_grid(coordinates, _dataset)
        for name in _moments:
            interpolated_data[name] = (
                interpolated_data[name] / interpolated_data[NAME_E]
            )

        return FrequencySpectrum(interpolated_data)

    def as_frequency_direction_spectrum(
        self,
        number_of_directions,
        method: Estimators = "mem2",
        solution_method="scipy",
    ) -> "FrequencyDirectionSpectrum":

        direction = numpy.linspace(0, 360, number_of_directions, endpoint=False)

        output_array = (
            estimate_directional_distribution(
                self.a1.values,
                self.b1.values,
                self.a2.values,
                self.b2.values,
                direction,
                method=method,
                solution_method=solution_method,
            )
            * self.e.values[..., None]
        )

        dims = self.dims_space_time + [NAME_F, NAME_D]
        coords = {x: self.dataset[x].values for x in self.dims}
        coords[NAME_D] = direction

        data = {NAME_E: (dims, output_array)}
        for x in self.dataset:
            if x in SPECTRAL_VARS:
                continue
            data[x] = (self.dims_space_time, self.dataset[x].values)

        return FrequencyDirectionSpectrum(Dataset(data_vars=data, coords=coords))


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
    dims=(NAME_T, NAME_F),
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
        NAME_T: numpy.atleast_1d(to_datetime64(time)),
        NAME_LAT: numpy.atleast_1d(latitude),
        NAME_LON: numpy.atleast_1d(longitude),
        NAME_DEPTH: numpy.atleast_1d(depth),
        NAME_F: frequency,
        NAME_E: variance_density,
        NAME_a1: a1,
        NAME_b1: b1,
        NAME_a2: a2,
        NAME_b2: b2,
    }

    return FrequencySpectrum(create_spectrum_dataset(dims, variables))


def create_2d_spectrum(
    frequency: numpy.ndarray,
    direction: numpy.ndarray,
    variance_density: numpy.ndarray,
    time,
    latitude: Union[numpy.ndarray, float],
    longitude: Union[numpy.ndarray, float],
    dims=(NAME_T, NAME_F, NAME_D),
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
        NAME_T: numpy.atleast_1d(to_datetime64(time)),
        NAME_LAT: numpy.atleast_1d(latitude),
        NAME_LON: numpy.atleast_1d(longitude),
        NAME_DEPTH: numpy.atleast_1d(depth),
        NAME_F: frequency,
        NAME_D: direction,
        NAME_E: variance_density,
    }
    return FrequencyDirectionSpectrum(create_spectrum_dataset(dims, variables))


def create_spectrum_dataset(dims, variables) -> Dataset:
    independent_variables = []
    for dim in dims:
        if dim in variables:
            independent_variables.append(dim)

    dependent_variables = [x for x in variables if x not in independent_variables]

    spectral_coords = {k: variables[k] for k in independent_variables}
    spatial_coords = {
        k: variables[k] for k in independent_variables if k not in SPECTRAL_DIMS
    }

    dataset = Dataset()
    for variable in dependent_variables:
        if variable in SPECTRAL_VARS:
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


def load_spectrum_from_netcdf(
    filename_or_obj,
) -> Union[FrequencySpectrum, FrequencyDirectionSpectrum]:
    """
    Load a spectrum from netcdf file
    :param filename_or_obj:
    :return:
    """
    dataset = open_dataset(filename_or_obj=filename_or_obj)
    if NAME_D in dataset.coords:
        return FrequencyDirectionSpectrum(dataset=dataset)
    else:
        return FrequencySpectrum(dataset=dataset)
