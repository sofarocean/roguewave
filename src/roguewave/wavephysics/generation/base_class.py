from roguewave import (
    FrequencyDirectionSpectrum,
    integrate_spectral_data,
)
from roguewave.wavespectra.spectrum import NAME_F, NAME_D
from xarray import DataArray
from abc import ABC
from typing import Literal

TWindInputType = Literal["u10", "friction_velocity", "ustar"]


class WindGeneration(ABC):
    def __init__(self, **kwargs):
        pass

    def rate(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        pass

    def bulk_rate(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        return integrate_spectral_data(
            self.rate(
                speed, direction, spectrum, roughness_length, wind_speed_input_type
            ),
            dims=[NAME_F, NAME_D],
        )

    def roughness(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length_guess: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        pass

    def u10_from_bulk_rate(
        self,
        bulk_rate: DataArray,
        guess_u10: DataArray,
        guess_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        pass
