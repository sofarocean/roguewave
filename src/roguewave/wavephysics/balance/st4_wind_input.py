from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    GRAVITATIONAL_ACCELERATION,
)

from roguewave.wavephysics.balance import WindGeneration
from numpy import cos, pi, log, exp, empty, inf
from numba import njit
from roguewave.wavetheory import inverse_intrinsic_dispersion_relation
from typing import TypedDict


class ST4WaveGenerationParameters(TypedDict):
    gravitational_acceleration: float
    charnock_maximum_roughness: float
    charnock_constant: float
    air_density: float
    water_density: float
    vonkarman_constant: float
    wave_age_tuning_parameter: float
    growth_parameter_betamax: float
    elevation: float


class ST4WindInput(WindGeneration):
    name = "st4 generation"

    def __init__(self, parameters: ST4WaveGenerationParameters = None):
        super(ST4WindInput, self).__init__(parameters)
        self._wind_source_term_function = _st4_wind_generation_point

    @staticmethod
    def default_parameters() -> ST4WaveGenerationParameters:
        return ST4WaveGenerationParameters(
            wave_age_tuning_parameter=0.006,
            growth_parameter_betamax=1.52,
            gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
            charnock_maximum_roughness=inf,
            charnock_constant=0.01,
            air_density=AIR.density,
            water_density=WATER.density,
            vonkarman_constant=AIR.vonkarman_constant,
            elevation=10,
        )


# ----------------------------------------------------------------------------------------------------------------------
# ST4 Point wise implementation functions (apply to a single spatial point)
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _st4_wind_generation_point(
    variance_density,
    wind,
    depth,
    roughness_length,
    st4_spectral_grid,
    st4_parameters,
    wind_source=None,
):
    """
    Implememtation of the st4 wind growth rate term

    :param variance_density:
    :param wind:
    :param depth:
    :param roughness_length:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """

    # Get the spectral size
    number_of_frequencies, number_of_directions = variance_density.shape

    # Allocate the output variable
    if wind_source is None:
        wind_source = empty((number_of_frequencies, number_of_directions))

    # Unpack variables passed in dictionaries/tuples for ease of reference and some slight speed benifits (avoid dictionary
    # lookup in loops).
    vonkarman_constant = st4_parameters["vonkarman_constant"]
    growth_parameter_betamax = st4_parameters["growth_parameter_betamax"]
    air_density = st4_parameters["air_density"]
    water_density = st4_parameters["water_density"]
    wave_age_tuning_parameter = st4_parameters["wave_age_tuning_parameter"]
    gravitational_acceleration = st4_parameters["gravitational_acceleration"]
    radian_frequency = st4_spectral_grid["radian_frequency"]
    radian_direction = st4_spectral_grid["radian_direction"]
    elevation = st4_parameters["elevation"]

    wind_forcing, wind_direction_degrees, wind_forcing_type = wind

    # Preprocess winds to the correct conventions (U10->friction velocity if need be, wind direction degrees->radians)
    if wind_forcing_type == "u10":
        friction_velocity = (
            wind_forcing * vonkarman_constant / log(elevation / roughness_length)
        )

    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        friction_velocity = wind_forcing

    else:
        raise ValueError("Unknown wind input type")

    wind_direction_radian = wind_direction_degrees * pi / 180

    # precalculate the constant multiplication.
    constant_factor = (
        growth_parameter_betamax / vonkarman_constant**2 * air_density / water_density
    )

    if depth == inf:
        wavenumber = radian_frequency**2 / gravitational_acceleration
    else:
        wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)

    cosine = cos(radian_direction - wind_direction_radian)

    # Loop over all frequencies/directions
    for frequency_index in range(number_of_frequencies):

        # Since wavenumber and wavespeed only depend on frequency we calculate those outside of the direction loop.
        constant_frequency_factor = constant_factor * radian_frequency[frequency_index]
        relative_speed = (
            friction_velocity
            * wavenumber[frequency_index]
            / radian_frequency[frequency_index]
        )

        for direction_index in range(number_of_directions):
            if cosine[direction_index] > 0:

                # Calculate the directional factor in the wind input formulation
                W = relative_speed * cosine[direction_index]

                # If the wave direction has an along wind component we continue.
                effective_wave_age = log(
                    wavenumber[frequency_index] * roughness_length
                ) + vonkarman_constant / (W + wave_age_tuning_parameter)

                if effective_wave_age > 0:
                    effective_wave_age = 0

                growth_rate = (
                    constant_frequency_factor
                    * exp(effective_wave_age)
                    * effective_wave_age**4
                    * W**2
                )

                wind_source[frequency_index, direction_index] = (
                    growth_rate * variance_density[frequency_index, direction_index]
                )
            else:
                wind_source[frequency_index, direction_index] = 0.0

    return wind_source
