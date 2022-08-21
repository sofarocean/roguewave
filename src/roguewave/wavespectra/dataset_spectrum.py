from xarray import Dataset, DataArray
from xarray.core.coordinates import DatasetCoordinates
from roguewave.tools.math import wrapped_difference
import numpy
from typing import Iterator, Hashable, Tuple
from roguewave.interpolate.dataset import interpolate_dataset_grid, \
    interpolate_dataset_along_axis
from roguewave.wavetheory.lineardispersion import \
    inverse_intrinsic_dispersion_relation
from roguewave.tools.time import to_datetime64

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

    def __setitem__(self, key, value):
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
        return numpy.argmax(self.e.where(range, 0).argmax(dim='frequency'))

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
        range = {'frequency': self._range(fmin, fmax)}

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

    def __add__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum['variance_density'] = (spectrum['variance_density'] +
                                        other['variance_density'])
        return spectrum

    def __sub__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum['variance_density'] = (spectrum['variance_density'] -
                                        other['variance_density'])
        return spectrum

    def __neg__(self, other: "WaveSpectrum") -> "WaveSpectrum":
        spectrum = self.copy(deep=True)
        spectrum['variance_density'] = -spectrum['variance_density']
        return spectrum

    def interpolate(self, coordinates):
        self.dataset = interpolate_dataset_grid(
            coordinates, self.dataset
        )
        return self

    def interpolate_frequency(self, new_frequencies):
        self.dataset = interpolate_dataset_along_axis(new_frequencies,
                                                      self.dataset,
                                                      coordinate_name='frequency')
        return self


class WaveFrequencySpectrum2D(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(WaveFrequencySpectrum2D, self).__init__(dataset)
        for name in _NAMES_2D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(f'Required variable/coordinate {name} is'
                                 f' not specified in the dataset')

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
        return (data_array * self.direction_step()).sum('direction')

    @property
    def e(self) -> DataArray:
        return self._directionally_integrate(self['variance_density'])

    @property
    def a1(self) -> DataArray:
        return self._directionally_integrate(
            self['variance_density'] * numpy.cos(self.radian_direction)
        )

    @property
    def b1(self) -> DataArray:
        return self._directionally_integrate(
            self['variance_density'] * numpy.sin(self.radian_direction)
        )

    @property
    def a2(self) -> DataArray:
        return self._directionally_integrate(
            self['variance_density'] * numpy.cos(2 * self.radian_direction)
        )

    @property
    def b2(self) -> DataArray:
        return self._directionally_integrate(
            self['variance_density'] * numpy.cos(
                2 * self.radian_direction)
        )

    @property
    def direction(self) -> DataArray:
        return self.dataset['direction']

    def spectrum_1d(self) -> "WaveFrequencySpectrum1D":
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


class WaveFrequencySpectrum1D(WaveSpectrum):
    def __init__(self, dataset: Dataset):
        super(WaveFrequencySpectrum1D, self).__init__(dataset)
        for name in _NAMES_1D:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(f'Required variable/coordinate {name} is'
                                 f' not specified in the dataset')


def create_1d_spectrum(frequency: numpy.ndarray,
                       variance_density: numpy.ndarray,
                       time,
                       latitude,
                       longitude,
                       a1=None,
                       b1=None,
                       a2=None,
                       b2=None,
                       depth=None) -> WaveFrequencySpectrum1D:
    time = numpy.atleast_1d(time)
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

    dataset = Dataset(
        data_vars={
            "variance_density": ((time, frequency), variance_density),
            "a1": ((time, frequency), a1),
            "b1": ((time, frequency), b1),
            "a2": ((time, frequency), a2),
            "b2": ((time, frequency), b2),
            "latitude": ((time,), latitude),
            "longitude": ((time,), longitude),
            "depth": ((time,), depth)
        },
        coords={'time': to_datetime64(time), "frequency": frequency}
    )
    return WaveFrequencySpectrum1D(dataset)


def create_2d_spectrum(frequency: numpy.ndarray,
                       direction: numpy.ndarray,
                       variance_density: numpy.ndarray,
                       time,
                       latitude,
                       longitude,
                       depth=None) -> WaveFrequencySpectrum2D:
    time = numpy.atleast_1d(time)
    latitude = numpy.atleast_1d(latitude)
    longitude = numpy.atleast_1d(longitude)
    if depth is None:
        depth = numpy.zeros(len(time)) + numpy.inf

    if variance_density.ndim == 2:
        variance_density = variance_density[None, :, :]

    dataset = Dataset(
        data_vars={
            "variance_density": (
            (time, frequency, direction), variance_density),
            "latitude": ((time,), latitude),
            "longitude": ((time,), longitude),
            "depth": ((time,), depth)
        },
        coords={'time': to_datetime64(time),
                "frequency": frequency,
                "direction": direction}
    )
    return WaveFrequencySpectrum2D(dataset)
