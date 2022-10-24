from roguewave.wavephysics.momentumflux.roughness_charnock import CharnockConstant
from roguewave.wavephysics.momentumflux.roughness_base_class import RoughnessLength
from roguewave.wavephysics.momentumflux.wavesupportedstress import wave_supported_stress
from roguewave.wavephysics.fluidproperties import (
    GRAVITATIONAL_ACCELERATION,
    FluidProperties,
    AIR,
    WATER,
)
from roguewave.wavespectra import (
    WaveSpectrum,
    FrequencyDirectionSpectrum,
)
from roguewave.wavephysics.generation import (
    create_wind_source_term,
    wind_parametrizations,
)
from roguewave.tools.solvers import fixed_point_iteration
from xarray import DataArray
from numpy import sqrt, log, inf


class Janssen(RoughnessLength):
    def __init__(
        self,
        charnock_constant=0.018,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
        base_parametrization: RoughnessLength = None,
        wind_generation: wind_parametrizations = "st4",
    ):
        super(Janssen, self).__init__()
        if base_parametrization is None:
            self._charnock = CharnockConstant(
                charnock_constant=charnock_constant,
                gravitational_acceleration=gravitational_acceleration,
            )
        else:
            self._charnock = base_parametrization

        self.generation = create_wind_source_term(wind_generation)
        self.gravitational_acceleration = gravitational_acceleration

    def form_drag(
        self,
        friction_velocity,
        spectrum: WaveSpectrum,
        direction_degrees,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
    ) -> DataArray:

        if direction_degrees is None:
            raise ValueError("direction must be specified")

        if isinstance(spectrum, FrequencyDirectionSpectrum):
            roughness_length_guess = self._charnock.form_drag(
                friction_velocity, spectrum, direction_degrees
            )
            spectrum: FrequencyDirectionSpectrum

            def iteration_function(roughness_length):
                wave_stress = wave_supported_stress(
                    spectrum,
                    friction_velocity,
                    direction_degrees,
                    roughness_length,
                    self.generation,
                    "friction_velocity",
                    air,
                    water,
                    self.gravitational_acceleration,
                )
                stress_ratio = abs(wave_stress / (air.density * friction_velocity**2))
                return self._charnock.form_drag(
                    friction_velocity, spectrum, direction_degrees
                ) / sqrt(1 - stress_ratio)

        else:
            raise ValueError("2D spectrum must be specified")

        return fixed_point_iteration(
            iteration_function, roughness_length_guess, bounds=(0, inf)
        )

    def roughness_from_speed(
        self,
        speed,
        elevation,
        spectrum: WaveSpectrum,
        direction_degrees=None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        **kwargs
    ) -> DataArray:

        if direction_degrees is None:
            raise ValueError("direction must be specified")

        if not isinstance(spectrum, FrequencyDirectionSpectrum):
            raise ValueError("2D spectrum must be specified")

        friction_velocity_guess = sqrt(self.drag_coefficient_estimate(speed)) * speed
        roughness_length_guess = self._charnock.roughness(
            friction_velocity_guess, spectrum, air, direction_degrees
        )

        def iteration_function(roughness_length):
            friction_velocity = (
                speed * air.vonkarman_constant / log(elevation / roughness_length)
            )
            wave_stress = wave_supported_stress(
                spectrum,
                friction_velocity,
                direction_degrees,
                roughness_length,
                self.generation,
                "friction_velocity",
                air,
                water,
                self.gravitational_acceleration,
            )
            stress_ratio = wave_stress / (air.density * friction_velocity**2)

            return self._charnock.form_drag(
                friction_velocity, spectrum, direction_degrees
            ) / sqrt(1 - stress_ratio) + self.smooth(friction_velocity, air)

        return fixed_point_iteration(
            iteration_function, roughness_length_guess, bounds=(0, inf)
        )

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

        friction_velocity = (
            speed * air.vonkarman_constant / log(elevation / roughness_length)
        )
        return wave_supported_stress(
            spectrum,
            friction_velocity,
            direction_degrees,
            roughness_length,
            self.generation,
            "friction_velocity",
            air,
            water,
            self.gravitational_acceleration,
        )

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
