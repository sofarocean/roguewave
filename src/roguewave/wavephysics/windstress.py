from roguewave import WaveSpectrum
from xarray import DataArray
from .roughnesslength import RoughnessLength
from .fluidproperties import AIR, WATER, FluidProperties, GRAVITATIONAL_ACCELERATION
from .windsource import wind_source_term
from .loglaw import friction_velocity_from_windspeed


def windstress_from_loglaw(
    windspeed: DataArray,
    elevation: float,
    spectrum: WaveSpectrum,
    roughness_length: RoughnessLength,
    air: FluidProperties = AIR,
) -> DataArray:
    """
    :param windspeed:
    :param elevation:
    :param spectrum:
    :param roughness_length:
    :param air:
    :return:
    """
    friction_velocity = friction_velocity_from_windspeed(
        windspeed, spectrum, roughness_length, elevation
    )
    return air.density * friction_velocity**2


def wave_supported_stress(
    spectrum: WaveSpectrum,
    u10: DataArray,
    wind_direction=0,
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
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
    wind_input = wind_source_term(spectrum, u10, wind_direction, air, water)
    return (
        water.density
        * gravitational_acceleration
        * (wind_input / spectrum.wave_speed()).integrate(coord="frequency")
    )


def viscous_supported_stress(u10: DataArray, air: FluidProperties = AIR) -> DataArray:
    """
    Banner and Peirson 1998
    :param u10:
    :return:
    """
    cd = 1.1e-3
    return air.density * cd * u10**2


def surface_stress(
    spectrum: WaveSpectrum,
    u10,
    wind_direction=0,
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
):

    return wave_supported_stress(
        spectrum, u10, wind_direction, air, water, gravitational_acceleration
    ) + viscous_supported_stress(u10, air)
