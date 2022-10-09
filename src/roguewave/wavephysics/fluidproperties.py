from xarray import DataArray

GRAVITATIONAL_ACCELERATION = 9.81
WATER_DENSITY = 1024
WATER_TEMPERATURE_DEGREES_C = 15
AIR_TEMPERATURE_DEGREES_C = 15
AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY_WATER = 1.19e-6
KINEMATIC_VISCOSITY_AIR = 1.48 * 10**-5


class FluidProperties:
    def __init__(
        self, density: DataArray, temperature: DataArray, kinematic_viscosity: DataArray
    ):
        self.density = density
        self.temperature = temperature
        self.kinematic_viscocity = kinematic_viscosity


AIR = FluidProperties(
    density=DataArray(data=AIR_DENSITY),
    temperature=DataArray(data=AIR_TEMPERATURE_DEGREES_C),
    kinematic_viscosity=DataArray(data=KINEMATIC_VISCOSITY_AIR),
)

WATER = FluidProperties(
    density=DataArray(data=WATER_DENSITY),
    temperature=DataArray(data=WATER_TEMPERATURE_DEGREES_C),
    kinematic_viscosity=DataArray(data=KINEMATIC_VISCOSITY_WATER),
)
