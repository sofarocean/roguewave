from roguewave import WaveSpectrum, FrequencySpectrum, FrequencyDirectionSpectrum
from roguewave.wavephysics.fluidproperties import (
    AIR,
    FluidProperties,
    WATER,
    GRAVITATIONAL_ACCELERATION,
)
from roguewave.tools.solvers import fixed_point_iteration
from xarray import DataArray, zeros_like
from numpy import log, inf, sqrt, exp
from abc import ABC, abstractmethod


class RoughnessLength(ABC):
    def roughness(
        self,
        friction_velocity,
        spectrum: WaveSpectrum,
        fluid_properties: FluidProperties,
        direction_degrees=None,
    ) -> DataArray:
        if issubclass(type(spectrum), FrequencySpectrum):
            spectrum = spectrum.as_frequency_direction_spectrum(36)
        return self.smooth(friction_velocity, fluid_properties) + self.form_drag(
            friction_velocity, spectrum, direction_degrees
        )

    @abstractmethod
    def form_drag(
        self, friction_velocity, spectrum: WaveSpectrum, direction_degrees
    ) -> DataArray:
        pass

    def smooth(self, friction_velocity, fluid_properties: FluidProperties) -> DataArray:
        return 0.11 * fluid_properties.kinematic_viscosity / friction_velocity

    def roughness_from_speed(
        self,
        speed,
        elevation,
        spectrum: WaveSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        guess=None,
        **kwargs
    ) -> DataArray:
        if issubclass(type(spectrum), FrequencySpectrum):
            spectrum = spectrum.as_frequency_direction_spectrum(36)

        if guess is None:
            drag = sqrt(self.drag_coefficient_estimate(speed))
            guess = 10 / exp(air.vonkarman_constant / drag)

        def _func(roughness):
            friction_velocity = (
                air.vonkarman_constant * speed / log(elevation / roughness)
            )
            return self.roughness(friction_velocity, spectrum, air, direction_degrees)

        return fixed_point_iteration(
            _func, guess, bounds=(0, inf), caller="roughness_from_speed", **kwargs
        )

    def drag_coefficient(
        self,
        speed,
        elevation,
        spectrum: WaveSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        **kwargs
    ) -> DataArray:
        if issubclass(type(spectrum), FrequencySpectrum):
            spectrum = spectrum.as_frequency_direction_spectrum(36)

        roughness = self.roughness_from_speed(
            speed, elevation, spectrum, direction_degrees, air, water
        )
        friction_velocity = speed * air.vonkarman_constant / log(elevation / roughness)
        return (friction_velocity / speed) ** 2

    def implied_charnock(
        self,
        speed,
        elevation,
        spectrum: WaveSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        **kwargs
    ) -> DataArray:
        roughness = self.roughness_from_speed(
            speed, elevation, spectrum, direction_degrees, air, water, **kwargs
        )
        friction_velocity = air.vonkarman_constant * speed / log(elevation / roughness)
        return roughness * GRAVITATIONAL_ACCELERATION / friction_velocity**2

    def drag_coefficient_estimate(self, speed):
        """
        Wu 1982 estimate
        :param speed:
        :return:
        """
        return (0.8 + 0.065 * speed) / 1000

    def total_stress(
        self, speed, elevation, roughness_length, air: FluidProperties = AIR
    ):

        friction_velocity = (
            speed * air.vonkarman_constant / log(elevation / roughness_length)
        )
        return air.density * friction_velocity**2

    def wave_supported_stress(
        self,
        speed,
        elevation,
        roughness_length,
        spectrum: FrequencyDirectionSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        **kwargs
    ):
        return zeros_like(speed)

    def turbulent_stress(
        self,
        speed,
        elevation,
        roughness_length,
        spectrum: FrequencyDirectionSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
    ):
        return self.total_stress(
            speed, elevation, roughness_length
        ) - self.wave_supported_stress(
            speed, elevation, roughness_length, spectrum, direction_degrees, air, water
        )
