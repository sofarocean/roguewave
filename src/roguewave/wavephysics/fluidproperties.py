from xarray import DataArray
from typing import Union

GRAVITATIONAL_ACCELERATION = 9.81
WATER_DENSITY = 1024
WATER_TEMPERATURE_DEGREES_C = 15
AIR_TEMPERATURE_DEGREES_C = 15
AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY_WATER = 1.19e-6
KINEMATIC_VISCOSITY_AIR = 1.48 * 10**-5
VONKARMAN_CONSTANT = 0.4

_input_type = Union[DataArray, float]


class FluidProperties:
    def __init__(
        self,
        density: _input_type,
        temperature: _input_type,
        kinematic_viscosity: _input_type,
        vonkarman_constant: _input_type,
    ):
        self.density = density
        self.temperature = temperature
        self.kinematic_viscosity = kinematic_viscosity
        self.vonkarman_constant = vonkarman_constant


AIR = FluidProperties(
    density=AIR_DENSITY,
    temperature=AIR_TEMPERATURE_DEGREES_C,
    kinematic_viscosity=KINEMATIC_VISCOSITY_AIR,
    vonkarman_constant=VONKARMAN_CONSTANT,
)

WATER = FluidProperties(
    density=WATER_DENSITY,
    temperature=WATER_TEMPERATURE_DEGREES_C,
    kinematic_viscosity=KINEMATIC_VISCOSITY_WATER,
    vonkarman_constant=VONKARMAN_CONSTANT,
)
