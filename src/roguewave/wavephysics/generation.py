from .fluidproperties import AIR, WATER, FluidProperties, GRAVITATIONAL_ACCELERATION
from roguewave import (
    FrequencyDirectionSpectrum,
    integrate_spectral_data,
)
from roguewave.wavespectra.spectrum import NAME_F, NAME_D, SPECTRAL_DIMS
from numpy import tanh, cos, pi, sqrt, log, exp, all
from xarray import DataArray, where
from abc import ABC, abstractmethod
from typing import Literal

wind_parametrizations = Literal["st6", "st4"]


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
            dims=[NAME_F, NAME_D],
        )


# ------------------
#       ST4
# ------------------
class ST4(WindGeneration):
    def __init__(
        self,
        vonkarman_constant=0.4,
        charnock_constant=0.012,
        wave_age_tuning_parameter=0.006,
        growth_parameter_betamax=0.5,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
        **kwargs,
    ):
        super(ST4, self).__init__(**kwargs)
        self.charnock_constant = charnock_constant
        self.vonkarman_constant = vonkarman_constant
        self.wave_age_tuning_parameter = wave_age_tuning_parameter
        self.growth_parameter_betamax = growth_parameter_betamax
        self.gravitational_acceleration = GRAVITATIONAL_ACCELERATION

    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
        memoize=False,
    ) -> DataArray:
        """

        :param u10:
        :param direction:
        :param spectrum:
        :param air:
        :param water:
        :param memoize:
        :return:
        """
        wave_speed = spectrum.wave_speed()
        wavenumber = spectrum.wavenumber
        roughness_length = 1e-6
        old_friction_velocity = 0
        updated_friction_velocity = (
            u10 / self.vonkarman_constant / log(10 / roughness_length)
        )
        for _iter in range(0, 30):
            # print(_iter)

            wind_input = st4_wind_source_term_ustar(
                spectrum.variance_density,
                spectrum.radian_frequency,
                updated_friction_velocity,
                direction,
                spectrum.direction,
                self.vonkarman_constant,
                wavenumber,
                wave_speed,
                roughness_length,
                self.wave_age_tuning_parameter,
                self.growth_parameter_betamax,
                water,
                air,
            )
            wave_stress = wave_supported_stress(
                wind_input, direction, spectrum.direction, wave_speed, water
            )
            roughness_length = self._roughness_length(
                updated_friction_velocity, wave_stress, air
            )
            old_friction_velocity = updated_friction_velocity
            updated_friction_velocity = (
                u10 * self.vonkarman_constant / log(10 / roughness_length)
            )
            print(
                _iter,
                (updated_friction_velocity - old_friction_velocity).values,
                updated_friction_velocity,
                roughness_length,
            )
            if all(abs(updated_friction_velocity - old_friction_velocity) < 1e-2):
                break
        else:
            raise ValueError("no convergence")

        return wind_input

    def z0(self, friction_velocity):
        return (
            self.charnock_constant
            * friction_velocity**2
            / self.gravitational_acceleration
        )

    def _roughness_length(
        self, friction_velocity, wave_stress, air: FluidProperties = AIR
    ):
        stress_ratio = wave_stress / (air.density * friction_velocity**2)
        return self.z0(friction_velocity) / sqrt(1 - stress_ratio)

    def roughness_length(
        self,
        friction_velocity: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
    ) -> DataArray:
        """

        :param u10:
        :param direction:
        :param spectrum:
        :param air:
        :param water:
        :param memoize:
        :return:
        """
        wave_speed = spectrum.wave_speed()
        wavenumber = spectrum.wavenumber
        roughness_length = 0.01 * self.z0(friction_velocity)
        for iter in range(0, 10):
            wind_input = st4_wind_source_term_ustar(
                spectrum.variance_density,
                spectrum.radian_frequency,
                friction_velocity,
                direction,
                spectrum.direction,
                self.vonkarman_constant,
                wavenumber,
                wave_speed,
                roughness_length,
                self.wave_age_tuning_parameter,
                self.growth_parameter_betamax,
                water,
                air,
            )
            wave_stress = wave_supported_stress(
                wind_input, direction, spectrum.direction, wave_speed, water
            )
            roughness_length = self._roughness_length(
                friction_velocity, wave_stress, air
            )
        return roughness_length


def wave_supported_stress(
    wind_input: DataArray,
    wind_direction,
    wave_direction,
    wave_speed: DataArray,
    water: FluidProperties = WATER,
) -> DataArray:
    delta = ((wave_direction - wind_direction + 180.0) % 360.0 - 180.0) * pi / 180
    return (
        GRAVITATIONAL_ACCELERATION
        * water.density
        * integrate_spectral_data(
            cos(delta) * wind_input / wave_speed, dims=SPECTRAL_DIMS
        )
    )


def st4_effective_wave_age(
    wind_forcing_parameter,
    vonkarman_constant,
    wavenumber,
    roughness_length,
    wave_age_tuning_parameter,
):
    factor = wind_forcing_parameter
    data = where(
        factor > 0,
        log(wavenumber * roughness_length)
        + vonkarman_constant / (factor + wave_age_tuning_parameter),
        0,
    )
    return where(data <= 0, data, 0)


def st4_temporal_growth_rate_wave_energy(
    friction_velocity,
    wind_direction_degrees,
    wave_direction,
    vonkarman_constant,
    wavenumber,
    wavespeed,
    roughness_length,
    wave_age_tuning_parameter,
    growth_parameter_betamax,
):
    wind_forcing_parameter = st4_wind_forcing_parameter(
        wavespeed, wave_direction, friction_velocity, wind_direction_degrees
    )
    effective_wave_age = st4_effective_wave_age(
        wind_forcing_parameter,
        vonkarman_constant,
        wavenumber,
        roughness_length,
        wave_age_tuning_parameter,
    )

    return (
        growth_parameter_betamax
        / vonkarman_constant**2
        * exp(effective_wave_age)
        * effective_wave_age**4
        * wind_forcing_parameter**2
    )


def st4_wind_forcing_parameter(
    wave_speed, direction, friction_velocity, wind_direction_degrees
) -> DataArray:
    """ """
    delta = ((direction - wind_direction_degrees + 180.0) % 360.0 - 180.0) * pi / 180
    W = friction_velocity / wave_speed * cos(delta)
    return where(W > 0, W, 0)


def st4_wind_source_term_ustar(
    variance_density,
    radian_frequency,
    friction_velocity,
    wind_direction_degrees,
    wave_direction,
    vonkarman_constant,
    wavenumber,
    wavespeed,
    roughness_length,
    wave_age_tuning_parameter,
    growth_parameter_betamax,
    water: FluidProperties = WATER,
    air: FluidProperties = AIR,
) -> DataArray:

    temporal_growth_rate_wave_energy = st4_temporal_growth_rate_wave_energy(
        friction_velocity,
        wind_direction_degrees,
        wave_direction,
        vonkarman_constant,
        wavenumber,
        wavespeed,
        roughness_length,
        wave_age_tuning_parameter,
        growth_parameter_betamax,
    )

    # Sin growth term
    wind_input = (
        air.density
        / water.density
        * (variance_density * radian_frequency * temporal_growth_rate_wave_energy)
    )

    return wind_input


# def gamma():


# ------------------
#       ST6
# ------------------
class ST6(WindGeneration):
    def __init__(self, **kwargs):
        super(ST6, self).__init__(**kwargs)
        self._cg = None
        self._wavenumber = None
        self._e = None
        self._peak = None
        self._peak_wave_speed = None
        self._object = None

    def memoize(self, spectrum: FrequencyDirectionSpectrum, memoize):
        if (spectrum is not self._object) or not memoize:
            self._object = spectrum
            self.wave_speed = spectrum.wave_speed()
            self.peak_wave_speed = spectrum.peak_wave_speed()
            self.peak_angular_frequency = spectrum.peak_angular_frequency()
            self.saturation_spectrum = (
                spectrum.wavenumber**3 * spectrum.group_velocity * spectrum.e / 2 / pi
            )

    def rate(
        self,
        u10: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        air=AIR,
        water=WATER,
        memoize=True,
    ) -> DataArray:
        self.memoize(spectrum, memoize)
        return st6_wind_source_term(
            spectrum.variance_density,
            u10,
            direction,
            self.wave_speed,
            spectrum.direction,
            self.peak_wave_speed,
            self.peak_angular_frequency,
            spectrum.radian_frequency,
            self.saturation_spectrum,
            air,
            water,
        )


def create_wind_source_term(
    wind_parametrization: wind_parametrizations = "st6", **kwargs
):
    if wind_parametrization == "st6":
        return ST6(**kwargs)
    elif wind_parametrization == "st4":
        return ST4(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization: {wind_parametrization}")


def st6_wind_source_term(
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
    gamma = st6_temporal_growth_rate_wave_energy(
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


def st6_temporal_growth_rate_wave_energy(
    u10: DataArray,
    wind_direction: DataArray,
    wave_speed: DataArray,
    direction: DataArray,
    peak_wave_speed: DataArray,
    peak_angular_frequency: DataArray,
    radian_frequency: DataArray,
    saturation_spectrum: DataArray,
) -> DataArray:
    """
    :param u10:
    :param wind_direction:
    :param wave_speed:
    :param direction:
    :param peak_wave_speed:
    :param peak_angular_frequency:
    :param radian_frequency:
    :param saturation_spectrum:
    :param output:
    :return:
    """

    W_squared = (
        st6_wind_forcing_parameter(wave_speed, direction, u10, wind_direction) ** 2
    )
    sqrt_spectral_saturation = sqrt(
        st6_spectral_saturation(
            u10,
            peak_wave_speed,
            peak_angular_frequency,
            radian_frequency,
            saturation_spectrum,
        )
    )
    W_times_sqrt_spectral_saturation = W_squared * sqrt_spectral_saturation
    return (
        st6_sheltering_coefficient(W_times_sqrt_spectral_saturation)
        * W_times_sqrt_spectral_saturation
    )


def st6_directional_spreading_function(
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


def st6_spectral_saturation(
    u10, peak_wave_speed, peak_angular_frequency, radian_frequency, saturation_spectrum
) -> DataArray:
    """

    :param u10:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return saturation_spectrum * st6_directional_spreading_function(
        u10, peak_wave_speed, peak_angular_frequency, radian_frequency
    )


def st6_wind_forcing_parameter(
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


def st6_sheltering_coefficient(
    W_times_sqrt_spectral_saturation,
) -> DataArray:
    """
    :param u10:
    :param wind_direction:
    :param spectrum:
    :return:
    """
    return 2.8 - (1 + tanh(10 * W_times_sqrt_spectral_saturation - 11))
