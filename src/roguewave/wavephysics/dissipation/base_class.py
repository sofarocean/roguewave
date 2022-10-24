from roguewave import FrequencyDirectionSpectrum, integrate_spectral_data, SPECTRAL_DIMS
from abc import ABC, abstractmethod
from xarray import DataArray
from numpy import pi, cos, sin, arctan2
from typing import Literal


breaking_parametrization = Literal["st6", "st4"]


class WaveBreaking(ABC):
    @abstractmethod
    def rate(self, spectrum: FrequencyDirectionSpectrum, memoize=None) -> DataArray:
        pass

    def saturation_spectrum(self, spectrum, cg, k, memoize=None):
        if memoize is None:
            memoize = {}

        if "saturation_spectrum" not in memoize:
            memoize["saturation_spectrum"] = cg * spectrum.e * k**3 / 2 / pi
        return memoize["saturation_spectrum"]

    def bulk_rate(
        self, spectrum: FrequencyDirectionSpectrum, memoize=None
    ) -> DataArray:
        frequency: Literal[
            "frequency"
        ] = "frequency"  # To avoid issues with pycharm flagging the argument as not being a literal
        direction: Literal["direction"] = "direction"
        return integrate_spectral_data(
            self.rate(spectrum, memoize), dims=[frequency, direction]
        )

    def mean_direction_degrees(
        self, spectrum: FrequencyDirectionSpectrum, memoize=None
    ):
        if memoize is None:
            memoize = {}

        # Disspation weighted average wave number to guestimate the wind direction. Note dissipation is negative- hence the
        # minus signs.
        kx = -integrate_spectral_data(
            self.rate(spectrum, memoize)
            * spectrum.wavenumber
            * cos(spectrum.radian_direction),
            dims=SPECTRAL_DIMS,
        )
        ky = -integrate_spectral_data(
            self.rate(spectrum, memoize)
            * spectrum.wavenumber
            * sin(spectrum.radian_direction),
            dims=SPECTRAL_DIMS,
        )

        return (arctan2(ky, kx) * 180 / pi) % 360
