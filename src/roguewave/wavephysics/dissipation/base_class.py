from roguewave import FrequencyDirectionSpectrum, integrate_spectral_data, SPECTRAL_DIMS
from abc import ABC, abstractmethod
from xarray import DataArray
from numpy import pi, cos, sin, arctan2
from typing import Literal

breaking_parametrization = Literal["st6", "st4"]


class Dissipation(ABC):
    @abstractmethod
    def rate(self, spectrum: FrequencyDirectionSpectrum, memoize=None) -> DataArray:
        pass

    def bulk_rate(
        self, spectrum: FrequencyDirectionSpectrum, memoize=None
    ) -> DataArray:
        return integrate_spectral_data(self.rate(spectrum, memoize), dims=SPECTRAL_DIMS)

    def mean_direction_degrees(
        self, spectrum: FrequencyDirectionSpectrum, memoize=None
    ):
        if memoize is None:
            memoize = {}

        # Disspation weighted average wave number to guestimate the wind direction. Note dissipation is negative- hence the
        # minus signs.
        rate = self.rate(spectrum, memoize)
        kx = -integrate_spectral_data(
            rate * spectrum.wavenumber * cos(spectrum.radian_direction),
            dims=SPECTRAL_DIMS,
        )
        ky = -integrate_spectral_data(
            rate * spectrum.wavenumber * sin(spectrum.radian_direction),
            dims=SPECTRAL_DIMS,
        )

        return (arctan2(ky, kx) * 180 / pi) % 360
