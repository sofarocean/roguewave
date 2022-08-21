from xarray import Dataset, DataArray, concat
from xarray.core.coordinates import DatasetCoordinates
from roguewave.tools.math import wrapped_difference
import numpy
from typing import Iterator, Hashable, Iterable
from roguewave.interpolate.dataset import interpolate_dataset_grid, \
    interpolate_dataset_along_axis
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation
from roguewave.tools.time import to_datetime64
from roguewave.wavespectra.estimators import mem2
from roguewave.wavespectra.parametric import pierson_moskowitz_frequency

_NAME_F = 'frequency'
_NAME_D = 'direction'
_NAME_T = 'time'
_NAME_E = 'variance_density'
_NAME_a1 = 'a1'
_NAME_b1 = 'b1'
_NAME_a2 = 'a2'
_NAME_b2 = 'b2'
_NAME_LAT = 'latitude'
_NAME_LON = 'longitude'
_NAME_DEPTH = 'depth'
_NAMES_2D = (_NAME_F, _NAME_D, _NAME_T, _NAME_E, _NAME_LAT, _NAME_LON)
_NAMES_1D = (_NAME_F, _NAME_T, _NAME_E, _NAME_LAT, _NAME_LON,
             _NAME_a1, _NAME_b1, _NAME_a2, _NAME_b2)


class DatasetWrapper():
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def __setitem__(self, key, value) -> None:
        return self.dataset.__setitem__(key, value)

    def __copy__(self):
        cls = self.__class__
        return cls(self.dataset.copy())

    def copy(self, deep=True):
        if deep:
            return self.__deepcopy__({})
        else:
            return self.__copy__()

    def __deepcopy__(self, memodict):
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
    frequency_units = 'Hertz'
    angular_units = 'Degrees'
    spectral_density_units = 'm**2/Hertz'
    angular_convention = 'Wave travel direction (going-to), measured anti-clockwise from East'
    bulk_properties = (
        'm0', 'hm0', 'tm01', 'tm02', 'peak_period', 'peak_direction',
        'peak_directional_spread', 'mean_direction', 'mean_directional_spread',
        'peak_frequency',
        'peak_wavenumber', 'latitude', 'longitude', 'time',
        'is_sea_spectrum', 'is_swell_spectrum')

    def __init__(self, dataset: Dataset):
        super(WaveSpectrum, self).__init__(dataset)

    def frequency_moment(self, power: int, fmin=0, fmax=numpy.inf
                         ) -> DataArray:
        _range = self._range(fmin, fmax)
        return (self.e.isel({_NAME_F: _range}) * self.frequency[
            _range] ** power
                ).integrate(coord=_NAME_F)

    def _range(self, fmin=0.0, fmax=numpy.inf) -> numpy.ndarray:
        return ((self[_NAME_F].values >= fmin) &
                (self[_NAME_F].values < fmax))

    @property
    def number_of_frequencies(self):
        return len(self.frequency)

    @property
    def spectral_values(self):
        return self.dataset[_NAME_E]

    @property
    def radian_frequency(self) -> DataArray:
        data_array = self[_NAME_F] * 2 * numpy.pi
        data_array.name = 'radian_frequency'
        return data_array

    @property
    def latitude(self) -> DataArray:
        return self[_NAME_LAT]

    @property
    def longitude(self) -> DataArray:
        return self[_NAME_LON]

    @property
    def time(self) -> DataArray:
        return self[_NAME_T]

    @property
    def e(self) -> DataArray:
        return self[_NAME_E]

    @property
    def a1(self) -> DataArray:
        return self[_NAME_a1]

    @property
    def b1(self) -> DataArray:
        return self[_NAME_b1]

    @property
    def a2(self) -> DataArray:
        return self[_NAME_a2]

    @property
    def b2(self) -> DataArray:
        return self[_NAME_b2]

    @property
    def A1(self) -> DataArray:
        return self.a1 * self.e

    @property
    def B1(self) -> DataArray:
        return self.b1 * self.e

    @property
    def A2(self) -> DataArray:
        return self.a2 * self.e

    @property
    def B2(self) -> DataArray:
        return self.b2 * self.e

    @property
    def frequency(self) -> DataArray:
        return self[_NAME_F]

    def m0(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return self.frequency_moment(0, fmin, fmax)

    def m1(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return self.frequency_moment(1, fmin, fmax)

    def m2(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return self.frequency_moment(2, fmin, fmax)

    def hm0(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return 4 * numpy.sqrt(self.m0(fmin, fmax))

    def tm01(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return self.m0(fmin, fmax) / self.m1(fmin, fmax)

    def tm02(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return numpy.sqrt(self.m0(fmin, fmax) / self.m2(fmin, fmax))

    def peak_index(self, fmin=0, fmax=numpy.inf) -> DataArray:
        range = self._range(fmin, fmax)
        return numpy.argmax(self.e.where(range, 0).argmax(dim=_NAME_F))

    def peak_frequency(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return self[_NAME_F][self.peak_index(fmin, fmax)]

    def peak_period(self, fmin=0, fmax=numpy.inf) -> DataArray:
        return 1 / self.peak_frequency(fmin, fmax)

    def peak_direction(self, fmin=0, fmax=numpy.inf) -> DataArray:
        index = self.peak_index(fmin, fmax)
        return self._mean_direction(
            self.a1.isel(**{_NAME_F: index}),
            self.a1.isel(**{_NAME_F: index}))

    def peak_directional_spread(self, fmin=0, fmax=numpy.inf):
        index = self.peak_index(fmin, fmax)
        a1 = self.a1.isel(**{_NAME_F: index})
        b1 = self.a1.isel(**{_NAME_F: index})
        return self._spread(a1, b1)

    @staticmethod
    def _mean_direction(a1, b1):
        return numpy.arctan2(b1, a1) * 180 / numpy.pi

    @staticmethod
    def _spread(a1, b1):
        return numpy.sqrt(
            2 - 2 * numpy.sqrt(a1 ** 2 + b1 ** 2)) * 180 / numpy.pi

    @property
    def mean_direction_per_frequency(self):
        return self._mean_direction(self.a1, self.b1)

    @property
    def mean_spread_per_frequency(self):
        return self._spread(self.a1, self.b1)

    def _spectral_weighted(self, property: DataArray, fmin=0, fmax=numpy.inf):
        range = {_NAME_F: self._range(fmin, fmax)}

        return numpy.trapz(property.isel(**range) * self.e.isel(**range),
                           self.frequency[range]) / self.m0(fmin, fmax)

    def mean_direction(self, fmin=0, fmax=numpy.inf):
        return self._mean_direction(self.mean_a1(fmin, fmax),
                                    self.mean_b1(fmin, fmax))

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
    def depth(self):
        if _NAME_DEPTH in self.dataset:
            return self.dataset[_NAME_DEPTH]
        else:
            return numpy.inf

    def wavenumber(self, **kwargs):
        return inverse_intrinsic_dispersion_relation(self.radian_frequency,
                                                     self.depth, **kwargs)

    def peak_wavenumber(self):
        index = self.peak_index()
        return inverse_intrinsic_dispersion_relation(
            self.radian_frequency[index], self.depth)

    def bulk_variables(self) -> Dataset:
        dataset = Dataset()
        dataset['significant_waveheight'] = self.significant_waveheight
        dataset['mean_period'] = self.mean_period
        dataset['peak_period'] = self.peak_period()
        dataset['peak_direction'] = self.peak_direction()
        dataset['peak_directional_spread'] = self.peak_directional_spread()
        dataset['mean_direction'] = self.mean_direction()
        dataset['mean_directional_spread'] = self.mean_directional_spread()
        dataset['peak_frequency'] = self.peak_frequency()
        dataset['latitude'] = self.latitude
        dataset['longitude'] = self.longitude
        dataset['timestamp'] = self.time
        return dataset

    @property
    def significant_waveheight(self):
        return self.hm0()

    @property
    def mean_period(self):
        return self.tm01()

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum[_NAME_E] = (spectrum[_NAME_E] +
                             other[_NAME_E])
        return spectrum

    def __sub__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum[_NAME_E] = (spectrum[_NAME_E] -
                             other[_NAME_E])
        return spectrum

    def __neg__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum[_NAME_E] = -spectrum[_NAME_E]
        return spectrum

    def interpolate(self, coordinates):
        self.dataset = interpolate_dataset_grid(
            coordinates, self.dataset
        )
        return self

    def interpolate_frequency(self, new_frequencies):
        self.dataset = interpolate_dataset_along_axis(
            new_frequencies, self.dataset, coordinate_name='frequency')
        return self

    @property
    def is_sea_spectrum(self) -> DataArray:
        """
        Identify whether or not it is a sea partion. Use 1D method for 2D
        spectra in section 3 of:

        Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
        Spectral partitioning and identification of wind sea and swell.
        Journal of atmospheric and oceanic technology, 26(1), 107-122.

        :return: boolean indicting it is sea
        """
        peak_index = self.peak_index()
        peak_frequency = self.peak_frequency()

        peak_level = \
            self.e.isel(indexers={'frequency': peak_index.values})
        pm_value = pierson_moskowitz_frequency(peak_frequency,
                                               peak_frequency)
        return peak_level == pm_value

    @property
    def is_swell_spectrum(self) -> DataArray:
        return ~self.is_sea_spectrum

    @classmethod
    def concat_from_list(cls,
                         _list: Iterable["WaveSpectrum"],
                         dim='time') -> "WaveSpectrum":
        _data = [x.dataset for x in _list]
        return cls(concat(_data, dim=dim))


class FrequencyDirectionSpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencyDirectionSpectrum, self).__init__(dataset)
        for name in _NAMES_2D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(f'Required variable/coordinate {name} is'
                                 f' not specified in the dataset')

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-2]))

    def direction_step(self) -> DataArray:
        difference = wrapped_difference(
            numpy.diff(self.direction.values,
                       append=self.direction[0]), period=360
        )
        return DataArray(
            data=difference,
            coords={_NAME_D: self.direction.values},
            dims=[_NAME_D]
        )

    @property
    def radian_direction(self) -> DataArray:
        data_array = self[_NAME_D] * numpy.pi / 180
        data_array.name = 'radian_direction'
        return data_array

    def _directionally_integrate(self, data_array: DataArray) -> DataArray:
        return (data_array * self.direction_step()).sum(_NAME_D)

    @property
    def e(self) -> DataArray:
        return self._directionally_integrate(self[_NAME_E])

    @property
    def a1(self) -> DataArray:
        return self._directionally_integrate(
            self[_NAME_E] * numpy.cos(self.radian_direction)
        )

    @property
    def b1(self) -> DataArray:
        return self._directionally_integrate(
            self[_NAME_E] * numpy.sin(self.radian_direction)
        )

    @property
    def a2(self) -> DataArray:
        return self._directionally_integrate(
            self[_NAME_E] * numpy.cos(2 * self.radian_direction)
        )

    @property
    def b2(self) -> DataArray:
        return self._directionally_integrate(
            self[_NAME_E] * numpy.cos(
                2 * self.radian_direction)
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
            self.b2.values
        )

    @property
    def number_of_directions(self):
        return len(self.direction)


class FrequencySpectrum(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(FrequencySpectrum, self).__init__(dataset)
        for name in _NAMES_1D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(f'Required variable/coordinate {name} is'
                                 f' not specified in the dataset')

    def __len__(self):
        return int(numpy.prod(self.spectral_values.shape[:-1]))

    def as_frequency_direction_spectrum(
            self, number_of_directions, method='mem2'
    ) -> "FrequencyDirectionSpectrum":

        direction = numpy.linspace(0, 360, number_of_directions,
                                   endpoint=False)
        radian_direction = direction * numpy.pi / 180

        input_dims = [str(x) for x in self.spectral_values.dims]
        input_shape = self.spectral_values.shape
        output_shape = (*input_shape, number_of_directions)

        loop_elements = numpy.prod(input_shape[:-1])
        output_array = numpy.empty(
            (loop_elements, self.number_of_frequencies, number_of_directions))

        for index in range(loop_elements):
            indices = numpy.unravel_index(index, input_shape[:-1])
            indexers = {
                dim: num for dim, num in zip(input_dims[:-1], indices)
            }
            output_array[index, :, :] = mem2(
                radian_direction,
                self.a1.isel(indexers).values,
                self.b1.isel(indexers).values,
                self.a2.isel(indexers).values,
                self.b2.isel(indexers).values,
            ) * numpy.pi / 180 * self.e.isel(indexers).values[:, None]
        output_array = numpy.reshape(output_array, output_shape)

        return create_2d_spectrum(
            frequency=self.frequency.values,
            direction=direction,
            variance_density=output_array,
            time=self.time.values,
            latitude=self.latitude.values,
            longitude=self.longitude.values,
            depth=self.depth.values
        )


def create_1d_spectrum(
        frequency: numpy.ndarray,
        variance_density: numpy.ndarray,
        time,
        latitude,
        longitude,
        a1=None,
        b1=None,
        a2=None,
        b2=None,
        depth=None) -> FrequencySpectrum:
    time = to_datetime64(numpy.atleast_1d(time))
    latitude = numpy.atleast_1d(latitude)
    longitude = numpy.atleast_1d(longitude)
    if depth is None:
        depth = numpy.zeros(len(time)) + numpy.inf

    if a1 is None: a1 = numpy.nan + numpy.ones_like(variance_density)
    if b1 is None: b1 = numpy.nan + numpy.ones_like(variance_density)
    if a2 is None: a2 = numpy.nan + numpy.ones_like(variance_density)
    if b2 is None: b2 = numpy.nan + numpy.ones_like(variance_density)

    if variance_density.ndim == 1:
        variance_density = variance_density[None, :, ]
        a1 = a1[None, :]
        b1 = b1[None, :]
        a2 = a2[None, :]
        b2 = b2[None, :]

    return FrequencySpectrum(Dataset(
        data_vars={
            _NAME_E: ((_NAME_T, _NAME_F), variance_density),
            _NAME_a1: ((_NAME_T, _NAME_F), a1),
            _NAME_b1: ((_NAME_T, _NAME_F), b1),
            _NAME_a2: ((_NAME_T, _NAME_F), a2),
            _NAME_b2: ((_NAME_T, _NAME_F), b2),
            _NAME_LAT: ((_NAME_T,), latitude),
            _NAME_LON: ((_NAME_T,), longitude),
            _NAME_DEPTH: ((_NAME_T,), depth)
        },
        coords={_NAME_T: to_datetime64(time), _NAME_F: frequency}
        )
    )


def create_2d_spectrum(frequency: numpy.ndarray,
                       direction: numpy.ndarray,
                       variance_density: numpy.ndarray,
                       time,
                       latitude,
                       longitude,
                       depth=None) -> FrequencyDirectionSpectrum:
    time = numpy.atleast_1d(time)
    latitude = numpy.atleast_1d(latitude)
    longitude = numpy.atleast_1d(longitude)
    if depth is None:
        depth = numpy.zeros(len(time)) + numpy.inf

    if variance_density.ndim == 2:
        variance_density = variance_density[None, :, :]

    dataset = Dataset(
        data_vars={
            _NAME_E: (
                (_NAME_T, _NAME_F, _NAME_D), variance_density),
            _NAME_LAT: ((_NAME_T,), latitude),
            _NAME_LON: ((_NAME_T,), longitude),
            _NAME_DEPTH: ((_NAME_T,), depth)
        },
        coords={_NAME_T: to_datetime64(time),
                _NAME_F: frequency,
                _NAME_D: direction}
    )
    return FrequencyDirectionSpectrum(dataset)
