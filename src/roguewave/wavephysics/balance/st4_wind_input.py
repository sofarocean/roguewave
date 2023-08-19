from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    GRAVITATIONAL_ACCELERATION,
)

from roguewave.wavephysics.balance import WindGeneration
from roguewave.wavephysics.balance._numba_settings import numba_default
from numpy import cos, pi, log, exp, empty, inf
from numba import jit
from linearwavetheory import inverse_intrinsic_dispersion_relation
from typing import TypedDict
from roguewave.wavephysics.balance.wam_tail_stress import (
    tail_stress_parametrization_wam,
)


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
    air_viscosity: float
    viscous_stress_parameter: float


class ST4WindInput(WindGeneration):
    name = "st4 generation"

    def __init__(self, parameters: ST4WaveGenerationParameters = None):
        super(ST4WindInput, self).__init__(parameters)
        self._wind_source_term_function = _st4_wind_generation_point
        self._tail_stress_parametrization_function = tail_stress_parametrization_wam

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
            air_viscosity=AIR.kinematic_viscosity,
            viscous_stress_parameter=0.0,
        )


# ----------------------------------------------------------------------------------------------------------------------
# ST4 Point wise implementation functions (apply to a single spatial point)
# ----------------------------------------------------------------------------------------------------------------------


@jit(**numba_default)
def _st4_wind_generation_point(
    variance_density,
    wind,
    depth: float,
    roughness_length: float,
    spectral_grid,
    parameters,
    wind_source=None,
):
    """
    Implementation of the st4 wind growth rate term as proposed by Janssen 1991- and as described in the ST4 paper.

    :param variance_density: input numpy array of shape (nf,nd), with nf number of frequencies, nd number of directions,
        that contains variance densities (unit m**2/Hz/Degree).
    :param wind: Wind input tuple (speed, direction, type) that descripes the wind speed, direction and kind of wind
        input. The latter can be: "u10" or "friction_velocity".
    :param depth: Local water depth in meter.
    :param roughness_length: Roughness length in meter.
    :param spectral_grid: numba dictionary containing the spectral grid (frequencies/directions)
    :param parameters: numba dictionary containing the parameters of the st4 algorithm. See the st4 class for details.
    :param wind_source: optional numpy array that contains the return values. This helps to avoid reallocating memory
        if the routine is called often in successive fashion (e.g. during iterative solutions).

    :return: numpy array of shape (nf,nd) containing for each frequency/direction pair the wind input in
        (m**2/Hz/Degree/s)
    """

    # Get the spectral size
    number_of_frequencies, number_of_directions = variance_density.shape

    vonkarman_constant = parameters["vonkarman_constant"]
    radian_frequency = spectral_grid["radian_frequency"]

    wind_forcing, wind_direction_degrees, wind_forcing_type = wind

    cosine_mutual_angle_wind_waves = cos(
        (spectral_grid["radian_direction"] - wind_direction_degrees * pi / 180 + pi)
        % (2 * pi)
        - pi
    )

    # Allocate the output variable
    if wind_source is None:
        wind_source = empty((number_of_frequencies, number_of_directions))

    # Preprocess winds to the correct conventions (U10->friction velocity if need be, wind direction degrees->radians)
    if wind_forcing_type == "u10":
        friction_velocity = (
            wind_forcing
            * vonkarman_constant
            / log(parameters["elevation"] / roughness_length)
        )
    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        friction_velocity = wind_forcing

    else:
        raise ValueError("Unknown wind input type")

    # precalculate the constant multiplication.
    constant_factor = (
        parameters["growth_parameter_betamax"]
        / vonkarman_constant**2
        * parameters["air_density"]
        / parameters["water_density"]
    )

    if depth == inf:
        wavenumber = radian_frequency**2 / parameters["gravitational_acceleration"]
    else:
        wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)

    # wave age tuning times the directional cosine.
    cosine_wave_age_tuning = (
        parameters["wave_age_tuning_parameter"] * cosine_mutual_angle_wind_waves
    )

    # Loop over all frequencies/directions
    for frequency_index in range(number_of_frequencies):

        # Since wavenumber and wavespeed only depend on frequency we calculate those outside of the direction loop.
        relative_speed = (
            wavenumber[frequency_index]
            * friction_velocity
            / radian_frequency[frequency_index]
        )

        for direction_index in range(number_of_directions):
            # If the wave direction has an along wind component we continue.
            if cosine_mutual_angle_wind_waves[direction_index] > 0:

                # Calculate the directional factor in the wind input formulation
                W = relative_speed * cosine_mutual_angle_wind_waves[direction_index]

                # The logarithm of the approximate expression for the dimensionless critical height
                # (Ardhuin 2010, eq 20).
                log_dimensionless_critical_height = log(
                    wavenumber[frequency_index] * roughness_length
                ) + vonkarman_constant / (W + cosine_wave_age_tuning[direction_index])

                # if the dimensionless critical height > 1 we assume there is no more energy transfer (Janssen 1991),
                # his equation 18.
                if log_dimensionless_critical_height > 0:
                    log_dimensionless_critical_height = 0

                growth_rate = (
                    constant_factor
                    * exp(log_dimensionless_critical_height)
                    * log_dimensionless_critical_height**4
                    * W**2
                    * radian_frequency[frequency_index]
                )

                wind_source[frequency_index, direction_index] = (
                    growth_rate * variance_density[frequency_index, direction_index]
                )

            else:
                wind_source[frequency_index, direction_index] = 0.0

    return wind_source
