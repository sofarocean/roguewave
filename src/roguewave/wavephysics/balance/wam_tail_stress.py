from numba import njit
from numpy import log, exp, array, pi, cos, sin

from roguewave.wavephysics.balance.solvers import numba_newton_raphson

#
# ----
# Wam Cy47r1 Implementation
# ----
#
# IFS DOCUMENTATION – Cy47r1 Operational implementation 30 June 2020 - PART VII
#


@njit()
def log_dimensionless_critical_height(
    x, charnock_constant, vonkarman_constant, wave_age_tuning_parameter
):
    """

    Dimensionless Critical Height according to Janssen (see IFS Documentation).
    :param x:
    :param charnock_constant:
    :param vonkarman_constant:
    :param wave_age_tuning_parameter:
    :return:
    """
    return (
        log(charnock_constant)
        + 2 * x
        + vonkarman_constant / (exp(x) + wave_age_tuning_parameter)
    )


@njit()
def integrate_tail_frequency_distribution(
    lower_bound, effective_charnock, vonkarman_constant, wave_age_tuning_parameter
):
    """
    Integrate the tail of the distributions. We are integrating

    log(Z(Y))Z(Y)**4 / Y      for   Y0 <= Y <= 1

    where

    Y = u_* / wavespeed
    Z = charnock * Y**2 * exp( vonkarman_constant / ( Y + wave_age_tuning_parameter)

    The boundaries of the integral are defined as the point where the critical height is at the surface
    (Y=1) and the point where Z >= 1 ( Y = Y0).

    We follow section 5 in the WAM documentation (see below). And introduce x = log(Y)

    so that we integrate in effect over

    log(Z(x))Z(x)**4     x0 <= x <= 0

    We find x0 as the point where Z(x0) = 0.

    REFERENCE:

    IFS DOCUMENTATION – Cy47r1 Operational implementation 30 June 2020 - PART VII

    :param effective_charnock:
    :param vonkarman_constant:
    :param wave_age_tuning_parameter:
    :return:
    """

    args = (effective_charnock, vonkarman_constant, wave_age_tuning_parameter)

    # find the location of the lower boundary of the integration domain. THis is where
    # loglog_mu = 0
    x0 = numba_newton_raphson(
        log_dimensionless_critical_height, log(0.01), args, (-10, 0), verbose=False
    )

    log_lower_bound = log(lower_bound)

    if log_lower_bound > x0:
        x0 = log_lower_bound

    if x0 > 0.0:
        return 0.0

    # Define the stepsize of the integration. Since the upper boundary is 0, this is merely the start
    stepsize = -x0 / 4

    # We use Boole's rule for integration.
    evaluation_points = array([x0, 3 * x0 / 4, 2 * x0 / 4, x0 / 4, 0])
    log_waveage = log_dimensionless_critical_height(evaluation_points, *args)
    values = log_waveage**4 * exp(log_waveage)
    integrant = (
        2
        / 45
        * stepsize
        * (
            7 * values[0]
            + 32 * values[1]
            + 12 * values[2]
            + 32 * values[3]
            + 7 * values[4]
        )
    )
    return integrant


@njit()
def tail_stress_parametrization_wam(
    variance_density,
    wind,
    depth,
    roughness_length,
    spectral_grid,
    parameters,
):
    vonkarman_constant = parameters["vonkarman_constant"]
    growth_parameter_betamax = parameters["growth_parameter_betamax"]
    wave_age_tuning_parameter = parameters["wave_age_tuning_parameter"]
    gravitational_acceleration = parameters["gravitational_acceleration"]
    radian_frequency = spectral_grid["radian_frequency"]
    radian_direction = spectral_grid["radian_direction"]
    elevation = parameters["elevation"]
    air_density = parameters["air_density"]

    number_of_frequencies, number_of_directions = variance_density.shape
    direction_step = spectral_grid["direction_step"]

    wind_forcing, wind_direction_degrees, wind_forcing_type = wind
    wind_direction_radian = wind_direction_degrees * pi / 180
    cosine_mutual_angle = cos(radian_direction - wind_direction_radian)
    cosine = cos(radian_direction)
    sine = sin(radian_direction)

    if wind_forcing_type == "u10":
        friction_velocity = (
            wind_forcing * vonkarman_constant / log(elevation / roughness_length)
        )

    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        friction_velocity = wind_forcing

    else:
        raise ValueError("Unknown wind input type")

    directional_integral_last_bin_east = 0.0
    directional_integral_last_bin_north = 0.0
    for direction_index in range(0, number_of_directions):
        if cosine_mutual_angle[direction_index] <= 0.0:
            continue

        directional_integral_last_bin_east += (
            cosine_mutual_angle[direction_index] ** 2
            * cosine[direction_index]
            * variance_density[number_of_frequencies - 1, direction_index]
            * direction_step[direction_index]
        )

        directional_integral_last_bin_north += (
            cosine_mutual_angle[direction_index] ** 2
            * sine[direction_index]
            * variance_density[number_of_frequencies - 1, direction_index]
            * direction_step[direction_index]
        )

    effective_charnock = (
        roughness_length * gravitational_acceleration / friction_velocity**2
    )

    lower_bound = (
        friction_velocity
        * radian_frequency[number_of_frequencies - 1]
        / gravitational_acceleration
    )
    frequency_integral = integrate_tail_frequency_distribution(
        lower_bound, effective_charnock, vonkarman_constant, wave_age_tuning_parameter
    )

    constant = (
        radian_frequency[number_of_frequencies - 1] ** 5
        / (2 * pi * gravitational_acceleration**2)
        * friction_velocity**2
        * growth_parameter_betamax
        / vonkarman_constant**2
    )

    # Add contribution of cappiliary waves
    total_stress = parameters["air_density"] * friction_velocity**2
    charnock_roughness = _charnock_relation_point(friction_velocity, parameters)
    background_stress = charnock_roughness**2 / roughness_length**2 * total_stress

    eastward_stress = (
        directional_integral_last_bin_east * frequency_integral * constant * air_density
        + cos(wind_direction_radian) * background_stress
    )

    northward_stress = (
        directional_integral_last_bin_north
        * frequency_integral
        * constant
        * air_density
        + sin(wind_direction_radian) * background_stress
    )

    return eastward_stress, northward_stress


@njit(cache=True)
def _charnock_relation_point(friction_velocity, parameters):
    """
    Charnock relation
    :param friction_velocity:
    :param parameters:
    :return:
    """
    roughness_length = (
        friction_velocity**2
        / parameters["gravitational_acceleration"]
        * parameters["charnock_constant"]
    )

    if roughness_length > parameters["charnock_maximum_roughness"]:
        roughness_length = parameters["charnock_maximum_roughness"]

    return roughness_length
