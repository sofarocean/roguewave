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
from xarray import DataArray
from numpy import log


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

        if not isinstance(spectrum, FrequencyDirectionSpectrum):
            raise ValueError("2D spectrum must be specified")

        return self.generation.roughness(
            friction_velocity,
            direction_degrees,
            spectrum,
            wind_speed_input_type="friction_velocity",
        )

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

        if direction_degrees is None:
            raise ValueError("direction must be specified")

        if not isinstance(spectrum, FrequencyDirectionSpectrum):
            raise ValueError("2D spectrum must be specified")

        return self.generation.roughness(
            speed, direction_degrees, spectrum, roughness_length_guess=guess
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
