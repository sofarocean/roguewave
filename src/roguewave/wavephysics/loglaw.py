from typing import Union
from roguewave import WaveSpectrum
from xarray import DataArray
from numpy.typing import ArrayLike
from roguewave.wavephysics.momentumflux import RoughnessLength
from roguewave.wavephysics.fluidproperties import FluidProperties, AIR, WATER
import numpy


def friction_velocity_from_speed(
    speed: Union[ArrayLike, DataArray],
    spectrum: WaveSpectrum,
    roughness_length: RoughnessLength,
    direction_degrees=None,
    elevation=10,
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    max_iter=100,
    rtol=1e-4,
    atol=1e-4,
):

    roughness_length = roughness_length.roughness_from_speed(
        speed,
        elevation,
        spectrum,
        direction_degrees,
        air,
        water,
        max_iter=max_iter,
        rtol=rtol,
        atol=atol,
    )
    return speed * air.vonkarman_constant / numpy.log(elevation / roughness_length)


def loglaw(
    friction_velocity,
    elevation,
    spectrum: WaveSpectrum,
    roughness_length: RoughnessLength,
    fluid_properties: FluidProperties,
    direction_degrees=None,
    max_iter=100,
    rtol=1e-7,
    atol=1e-7,
):
    return (
        friction_velocity
        / fluid_properties.vonkarman_constant
        * numpy.log(
            elevation
            / roughness_length.roughness(
                friction_velocity,
                spectrum,
                direction_degrees=direction_degrees,
                fluid_properties=fluid_properties,
            )
        )
    )
