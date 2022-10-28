from roguewave.wavephysics.generation import WindGeneration, TWindInputType
from roguewave.wavespectra import (
    FrequencyDirectionSpectrum,
    integrate_spectral_data,
    SPECTRAL_DIMS,
)
from xarray import DataArray
from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from numpy import cos, pi, sqrt, sin


def wave_supported_stress(
    spectrum: FrequencyDirectionSpectrum,
    speed: DataArray,
    wind_direction,
    roughness_length: DataArray,
    generation: WindGeneration,
    wind_input_type: TWindInputType = "u10",
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
    memoized=None,
) -> DataArray:
    """

    :param spectrum:
    :param u10:
    :param wind_direction:
    :param air:
    :param water:
    :param gravitational_acceleration:
    :return:
    """
    if memoized is None:
        memoized = {}

    if "wave_speed" not in memoized:
        memoized["wave_speed"] = spectrum.wave_speed()

    if "wind_mutual_angle" not in memoized:
        memoized["wind_mutual_angle"] = (
            ((spectrum.direction - wind_direction + 180.0) % 360.0 - 180.0) * pi / 180
        )

    wind_input = generation.rate(
        speed,
        wind_direction,
        spectrum,
        roughness_length,
        wind_input_type,
        air,
        water,
        memoized,
    )

    stress_downwind = (
        gravitational_acceleration
        * water.density
        * integrate_spectral_data(
            cos(memoized["wind_mutual_angle"]) * wind_input / memoized["wave_speed"],
            dims=SPECTRAL_DIMS,
        )
    )

    stress_crosswind = (
        gravitational_acceleration
        * water.density
        * integrate_spectral_data(
            sin(memoized["wind_mutual_angle"]) * wind_input / memoized["wave_speed"],
            dims=SPECTRAL_DIMS,
        )
    )

    total_stress = sqrt(stress_crosswind**2 + stress_downwind**2)
    return total_stress
