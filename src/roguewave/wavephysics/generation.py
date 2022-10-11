from .fluidproperties import AIR, WATER, FluidProperties, GRAVITATIONAL_ACCELERATION
from roguewave import (
    WaveSpectrum,
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
    integrate_spectral_data,
)
from numpy import tanh, cos, pi, sqrt
from xarray import DataArray, where
from abc import ABC, abstractmethod
from typing import Literal

wind_parametrizations = Literal["st6"]


class WindGeneration(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
    ) -> DataArray:
        pass

    def bulk_rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
    ) -> DataArray:
        return integrate_spectral_data(
            self.rate(u10, direction, spectrum, air, water),
            dims=["frequency", "direction"],
        )


class ST6(WindGeneration):
    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencySpectrum,
        air=AIR,
        water=WATER,
    ) -> DataArray:
        return wind_source_term(spectrum, u10, direction, air, water)


class ST4(WindGeneration):
    def __init__(self, growth_parameter, vonkarman_constant=0.4):
        self.growth_parameter = growth_parameter
        self.vonkarman_constant = vonkarman_constant

    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencySpectrum,
        air=AIR,
        water=WATER,
    ) -> DataArray:
        pass


def create_wind_source_term(
    wind_parametrization: wind_parametrizations = "st6", **kwargs
):
    if wind_parametrization == "st6":
        return ST6(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization: {wind_parametrization}")


def wind_source_term(
    spectrum: WaveSpectrum,
    u10,
    wind_direction=DataArray(data=0),
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
):
    """
    :param spectrum:
    :param u10:
    :param wind_direction:
    :param air:
    :param water:
    :return:
    """

    # Growth rate factor
    gamma = temporal_growth_rate_wave_energy(
        u10, wind_direction, spectrum, gravitational_acceleration
    )

    # Sin growth term
    wind_input = (
        air.density
        / water.density
        * (spectrum.variance_density * spectrum.radian_frequency * gamma)
    )

    return wind_input


def temporal_growth_rate_wave_energy(
    u10: DataArray,
    wind_direction: DataArray,
    spectrum: WaveSpectrum,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
) -> DataArray:
    """

    :param u10:
    :param wind_direction:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return (
        sheltering_coefficient(u10, wind_direction, spectrum)
        * sqrt(
            spectral_saturation(
                u10, spectrum, gravitational_acceleration=gravitational_acceleration
            )
        )
        * wind_forcing_parameter(spectrum, u10, wind_direction) ** 2
    )


def directional_spreading_function(u10: DataArray, spectrum: WaveSpectrum) -> DataArray:
    """
    Babanin & Soloviev, 1998.

    :param u10:
    :param spectrum:
    :return:
    """
    cp = spectrum.peak_wave_speed()
    peak_omega = spectrum.peak_angular_frequency()
    omega = spectrum.radian_frequency
    return 1.12 * (u10 / cp) ** (-0.5) * (omega / peak_omega) ** -(0.95) + 1 / (2 * pi)


def spectral_saturation(
    u10, spectrum: WaveSpectrum, gravitational_acceleration=9.81
) -> DataArray:
    """

    :param u10:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return (
        spectrum.group_velocity
        * spectrum.e
        * spectrum.wavenumber**3
        / 2
        / pi
        * directional_spreading_function(u10, spectrum)
    )


def wind_forcing_parameter(
    spectrum: WaveSpectrum, u10, wind_direction_degrees
) -> DataArray:
    """

    :param spectrum:
    :param u10:
    :param wind_direction_degrees:
    :return:
    """
    if isinstance(spectrum, FrequencySpectrum):
        W = u10 / spectrum.wave_speed() - 1

    elif isinstance(spectrum, FrequencyDirectionSpectrum):
        delta = (spectrum.direction - wind_direction_degrees + 180.0) % 360.0 - 180.0
        W = u10 * cos(delta * pi / 180) / spectrum.wave_speed() - 1

    else:
        raise ValueError("unknown spectral object")

    return where(W > 0, W, 0)


def sheltering_coefficient(
    u10: DataArray, wind_direction: DataArray, spectrum: WaveSpectrum
) -> DataArray:
    """

    :param u10:
    :param wind_direction:
    :param spectrum:
    :return:
    """
    W = wind_forcing_parameter(spectrum, u10, wind_direction)
    return 2.8 - (1 + tanh(10 * sqrt(spectral_saturation(u10, spectrum)) * W**2 - 11))
