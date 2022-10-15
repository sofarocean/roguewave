from .fluidproperties import AIR, WATER, FluidProperties, GRAVITATIONAL_ACCELERATION
from roguewave import (
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
        self._cg = None
        self._wavenumber = None
        self._e = None
        self._peak = None
        self._peak_wave_speed = None
        self._object = None

    def memoize(self, spectrum: FrequencyDirectionSpectrum):
        self.wave_speed = spectrum.wave_speed()
        self.peak_wave_speed = spectrum.peak_wave_speed()
        self.peak_angular_frequency = spectrum.peak_angular_frequency()
        self.saturation_spectrum = (
            spectrum.wavenumber**3 * spectrum.group_velocity * spectrum.e
        )

    @abstractmethod
    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
        memoize=False,
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
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
        memoize=True,
    ) -> DataArray:

        variance_density = spectrum.variance_density
        if not memoize:
            wave_speed = spectrum.wave_speed()
            peak_wave_speed = spectrum.peak_wave_speed()
            peak_angular_frequency = spectrum.peak_angular_frequency()
            saturation_spectrum = (
                spectrum.wavenumber**3 * spectrum.group_velocity * spectrum.e
            )
        else:
            if spectrum is not self._object:
                self.memoize(spectrum)

            wave_speed = self.wave_speed
            peak_wave_speed = self.peak_wave_speed
            peak_angular_frequency = self.peak_angular_frequency
            saturation_spectrum = self.saturation_spectrum

        return wind_source_term(
            variance_density,
            u10,
            direction,
            wave_speed,
            spectrum.direction,
            peak_wave_speed,
            peak_angular_frequency,
            spectrum.radian_frequency,
            saturation_spectrum,
            air,
            water,
        )


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
        memoize=False,
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
    spectrum,
    u10,
    wind_direction,
    wave_speed,
    direction,
    peak_wave_speed,
    peak_angular_frequency,
    radian_frequency,
    saturation_spectrum,
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
        u10,
        wind_direction,
        wave_speed,
        direction,
        peak_wave_speed,
        peak_angular_frequency,
        radian_frequency,
        saturation_spectrum,
    )

    # Sin growth term
    wind_input = air.density / water.density * (spectrum * radian_frequency * gamma)

    return wind_input


def temporal_growth_rate_wave_energy(
    u10: DataArray,
    wind_direction: DataArray,
    wave_speed,
    direction,
    peak_wave_speed,
    peak_angular_frequency,
    radian_frequency,
    saturation_spectrum,
) -> DataArray:
    """

    :param u10:
    :param wind_direction:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return (
        sheltering_coefficient(
            u10,
            wind_direction,
            wave_speed,
            direction,
            peak_wave_speed,
            peak_angular_frequency,
            radian_frequency,
            saturation_spectrum,
        )
        * sqrt(
            spectral_saturation(
                u10,
                peak_wave_speed,
                peak_angular_frequency,
                radian_frequency,
                saturation_spectrum,
            )
        )
        * wind_forcing_parameter(wave_speed, direction, u10, wind_direction) ** 2
    )


def directional_spreading_function(
    u10: DataArray, peak_wave_speed, peak_angular_frequency, radian_frequency
) -> DataArray:
    """
    Babanin & Soloviev, 1998.

    :param u10:
    :param spectrum:
    :return:
    """
    return 1.12 * (u10 / peak_wave_speed) ** (-0.5) * (
        radian_frequency / peak_angular_frequency
    ) ** -(0.95) + 1 / (2 * pi)


def spectral_saturation(
    u10, peak_wave_speed, peak_angular_frequency, radian_frequency, saturation_spectrum
) -> DataArray:
    """

    :param u10:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return (
        saturation_spectrum
        / 2
        / pi
        * directional_spreading_function(
            u10, peak_wave_speed, peak_angular_frequency, radian_frequency
        )
    )


def wind_forcing_parameter(
    wave_speed, direction, u10, wind_direction_degrees
) -> DataArray:
    """

    :param spectrum:
    :param u10:
    :param wind_direction_degrees:
    :return:
    """
    delta = (direction - wind_direction_degrees + 180.0) % 360.0 - 180.0
    W = u10 * cos(delta * pi / 180) / wave_speed - 1
    return where(W > 0, W, 0)


def sheltering_coefficient(
    u10: DataArray,
    wind_direction: DataArray,
    wave_speed,
    direction,
    peak_wave_speed,
    peak_angular_frequency,
    radian_frequency,
    saturation_spectrum,
) -> DataArray:
    """
    :param u10:
    :param wind_direction:
    :param spectrum:
    :return:
    """
    W = wind_forcing_parameter(wave_speed, direction, u10, wind_direction)
    return 2.8 - (
        1
        + tanh(
            10
            * sqrt(
                spectral_saturation(
                    u10,
                    peak_wave_speed,
                    peak_angular_frequency,
                    radian_frequency,
                    saturation_spectrum,
                )
            )
            * W**2
            - 11
        )
    )
