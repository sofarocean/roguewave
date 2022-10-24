from roguewave.wavephysics.fluidproperties import AIR, WATER, FluidProperties
from roguewave import (
    FrequencyDirectionSpectrum,
    integrate_spectral_data,
)
from roguewave.wavespectra.spectrum import NAME_F, NAME_D
from xarray import DataArray
from abc import ABC, abstractmethod
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
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:

        if wind_speed_input_type == "u10":
            return self.rate_U10(
                speed, direction, spectrum, roughness_length, air, water, memoized
            )

        elif wind_speed_input_type in ["ustar", "friction_velocity"]:
            return self.rate_friction_velocity(
                speed, direction, spectrum, roughness_length, air, water, memoized
            )

        else:
            raise ValueError(
                f"Unknown input type {wind_speed_input_type}, "
                f"has to be one of: 'u10','friction_velocity','ustar'"
            )

    def bulk_rate(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:
        return integrate_spectral_data(
            self.rate(
                speed,
                direction,
                spectrum,
                roughness_length,
                wind_speed_input_type,
                air,
                water,
                memoized,
            ),
            dims=[NAME_F, NAME_D],
        )

    @abstractmethod
    def rate_U10(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:
        pass

    @abstractmethod
    def rate_friction_velocity(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:
        pass
