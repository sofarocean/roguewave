from roguewave.wavephysics.balance.jb23_tail_stress import (
    tail_stress_parametrization_jb23,
)
from roguewave.wavephysics.balance.st4_wind_input import (
    ST4WindInput,
)
from roguewave.wavephysics.fluidproperties import GRAVITATIONAL_ACCELERATION, AIR, WATER

from numpy import cos, pi, log, exp, empty, inf, sin
from numba import njit
from linearwavetheory import inverse_intrinsic_dispersion_relation
from typing import TypedDict


class JB23WaveGenerationParameters(TypedDict):
    gravitational_acceleration: float
    charnock_maximum_roughness: float
    charnock_constant: float
    air_density: float
    water_density: float
    vonkarman_constant: float
    wave_age_tuning_parameter: float
    growth_parameter_betamax: float
    elevation: float
    air_viscosity: float
    viscous_stress_parameter: float
    surface_tension: float
    width_factor: float
    non_linear_effect_strength: float


class JB23WindInput(ST4WindInput):
    name = "JB23 generation"

    def __init__(self, parameters: JB23WaveGenerationParameters = None):
        super(JB23WindInput, self).__init__(parameters)
        # The obly difference with ST4 as implemented is the calculation of the tail contributions.
        self._wind_source_term_function = _jb23_wind_generation_point
        self._tail_stress_parametrization_function = tail_stress_parametrization_jb23

    @staticmethod
    def default_parameters() -> JB23WaveGenerationParameters:
        return JB23WaveGenerationParameters(
            wave_age_tuning_parameter=0.008,
            growth_parameter_betamax=2,
            gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
            charnock_maximum_roughness=inf,
            charnock_constant=0.005,
            air_density=AIR.density,
            water_density=WATER.density,
            vonkarman_constant=AIR.vonkarman_constant,
            elevation=10,
            air_viscosity=AIR.kinematic_viscosity,
            viscous_stress_parameter=1.0 / 25.0,
            surface_tension=WATER.kinematic_surface_tension,
            width_factor=0.6,
            non_linear_effect_strength=1.0,
        )


# ----------------------------------------------------------------------------------------------------------------------
# ST4 Point wise implementation functions (apply to a single spatial point)
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _jb23_wind_generation_point(
    variance_density,
    wind,
    depth,
    roughness_length,
    spectral_grid,
    parameters,
    wind_source=None,
):
    """
    Implememtation of the st4 wind growth rate term

    :param variance_density:
    :param wind:
    :param depth:
    :param roughness_length:
    :param spectral_grid:
    :param parameters:
    :return:
    """

    # Get the spectral size
    number_of_frequencies, number_of_directions = variance_density.shape

    # Allocate the output variable
    if wind_source is None:
        wind_source = empty((number_of_frequencies, number_of_directions))

    # Unpack variables passed in dictionaries/tuples for ease of reference and some slight speed benifits
    # (avoid dictionary lookup in loops).
    vonkarman_constant = parameters["vonkarman_constant"]
    growth_parameter_betamax = parameters["growth_parameter_betamax"]
    air_density = parameters["air_density"]
    water_density = parameters["water_density"]
    gravitational_acceleration = parameters["gravitational_acceleration"]
    radian_frequency = spectral_grid["radian_frequency"]
    radian_direction = spectral_grid["radian_direction"]
    direction_step = spectral_grid["direction_step"]
    elevation = parameters["elevation"]
    non_linear_effect_strength = parameters["non_linear_effect_strength"]

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

    mutual_angle = (radian_direction - wind_direction_radian + pi) % (2 * pi) - pi
    cosine = cos(mutual_angle)
    sine2 = sin(mutual_angle) ** 2

    # Loop over all frequencies/directions
    epsilon = air_density / water_density

    cosine_wave_age_tuning = parameters["wave_age_tuning_parameter"] * cosine
    for frequency_index in range(number_of_frequencies):

        # Since wavenumber and wavespeed only depend on frequency we calculate those outside of the direction loop.
        wavespeed = radian_frequency[frequency_index] / wavenumber[frequency_index]
        group_speed = wavespeed / 2

        relative_speed = friction_velocity / wavespeed

        N1 = 0.0
        N2 = 0.0

        for direction_index in range(number_of_directions):
            # If the wave direction has an along wind component we continue.
            if cosine[direction_index] > 0:

                # Calculate the directional factor in the wind input formulation
                W = relative_speed * cosine[direction_index]

                effective_wave_age = log(
                    wavenumber[frequency_index] * roughness_length
                ) + vonkarman_constant / (W + cosine_wave_age_tuning[direction_index])

                if effective_wave_age > 0:
                    effective_wave_age = 0

                growth_rate = (
                    constant_factor
                    * exp(effective_wave_age)
                    * effective_wave_age**4
                    * W**2
                    * radian_frequency[frequency_index]
                )

                # Note that Peter does not include the k factor into the defenition of his F(k,\theta). So we need
                # an additional division by k (in addition to the jacobian cg) to make this work-
                # hence the wavenumber **2 !!
                common_factor = (
                    non_linear_effect_strength
                    * variance_density[frequency_index, direction_index]
                    * group_speed
                    * direction_step[direction_index]
                    * wavenumber[frequency_index] ** 2
                    * growth_rate
                    / vonkarman_constant
                    / friction_velocity
                    / epsilon
                    / 2
                    / pi
                )

                N1 += common_factor * sine2[direction_index]
                N2 += common_factor

                wind_source[frequency_index, direction_index] = (
                    growth_rate * variance_density[frequency_index, direction_index]
                )
            else:
                wind_source[frequency_index, direction_index] = 0.0

        #
        for direction_index in range(number_of_directions):
            wind_source[frequency_index, direction_index] = (
                wind_source[frequency_index, direction_index] * (1 + N1) / (1 + N2)
            )

    return wind_source
