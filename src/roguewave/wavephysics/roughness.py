from roguewave import FrequencyDirectionSpectrum
from roguewave.wavephysics.fluidproperties import (
    AIR,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from roguewave.tools.solvers import fixed_point_iteration
from roguewave.wavephysics.balance import SourceTermBalance
from xarray import DataArray
from numpy import log, inf, sqrt, exp


def drag_coefficient_wu(speed):
    return (0.8 + 0.065 * speed) / 1000


def roughness_wu(speed, elevation, air: FluidProperties = AIR):
    drag = sqrt(drag_coefficient_wu(speed))
    return elevation / exp(air.vonkarman_constant / drag)


def charnock_roughness_length(friction_velocity: DataArray, **kwargs) -> DataArray:
    charnock_constant = kwargs.get("charnock_constant", 0.012)
    gravitational_acceleration = kwargs.get(
        "gravitational_acceleration", GRAVITATIONAL_ACCELERATION
    )

    return charnock_constant * friction_velocity**2 / gravitational_acceleration


def charnock_roughness_length_from_u10(speed, **kwargs) -> DataArray:
    air = kwargs.get("air", AIR)
    elevation = kwargs.get("elevation", 10)
    guess = kwargs.get("guess", roughness_wu(speed, elevation, air.vonkarman_constant))

    def _func(roughness):
        friction_velocity = air.vonkarman_constant * speed / log(elevation / roughness)
        return roughness(friction_velocity, **kwargs)

    return fixed_point_iteration(
        _func, guess, bounds=(0, inf), caller="roughness_from_speed", **kwargs
    )


def janssen_roughness_length(
    friction_velocity: DataArray,
    spectrum: FrequencyDirectionSpectrum,
    balance: SourceTermBalance,
    wind_direction: DataArray = None,
):
    if wind_direction is None:
        wind_direction = balance.dissipation.mean_direction_degrees(spectrum)

    return balance.generation.roughness(
        friction_velocity,
        direction=wind_direction,
        spectrum=spectrum,
        wind_speed_input_type="friction_velocity",
    )


def janssen_roughness_length_from_u10(
    friction_velocity: DataArray,
    spectrum: FrequencyDirectionSpectrum,
    balance: SourceTermBalance,
    wind_direction: DataArray = None,
    **kwargs
):
    if wind_direction is None:
        wind_direction = balance.dissipation.mean_direction_degrees(spectrum)

    return balance.generation.roughness(
        friction_velocity,
        direction=wind_direction,
        spectrum=spectrum,
        wind_speed_input_type="friction_velocity",
    )


def drag_coefficient(u10: DataArray, roughness: DataArray, **kwargs) -> DataArray:

    air = kwargs.get("air", AIR)
    elevation = kwargs.get("elevation", 10)

    friction_velocity = u10 * air.vonkarman_constant / log(elevation / roughness)
    return (friction_velocity / u10) ** 2
