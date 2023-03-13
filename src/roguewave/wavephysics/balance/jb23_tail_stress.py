from numba import njit
from numpy import log, inf, array, exp, empty, pi, cos, tanh, sqrt, linspace, sin

from roguewave.wavephysics.balance.solvers import numba_newton_raphson
from roguewave.wavephysics.balance.wam_tail_stress import (
    log_dimensionless_critical_height,
)


# --------------------------------
# Janssen and Bidlot 2022
# --------------------------------
@njit()
def tail_stress_parametrization_jb23(
    variance_density, wind, depth, roughness_length, spectral_grid, parameters
):
    wind_forcing, wind_direction_degrees, wind_forcing_type = wind
    wind_direction_radian = wind_direction_degrees * pi / 180
    cosine = cos(spectral_grid["radian_direction"] - wind_direction_radian)

    if wind_forcing_type == "u10":
        friction_velocity = (
            wind_forcing
            * parameters["vonkarman_constant"]
            / log(parameters["elevation"] / roughness_length)
        )

    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        friction_velocity = wind_forcing

    else:
        raise ValueError("Unknown wind input type")

    directional_integral_last_bin = 0
    direction_step = spectral_grid["direction_step"]
    number_of_frequencies, number_of_directions = variance_density.shape
    for direction_index in range(0, number_of_directions):
        if cosine[direction_index] <= 0.0:
            continue

        directional_integral_last_bin += (
            variance_density[number_of_frequencies - 1, direction_index]
            * direction_step[direction_index]
        )

    last_resolved_wavenumber = (
        spectral_grid["radian_frequency"][number_of_frequencies - 1] ** 2
        / parameters["gravitational_acceleration"]
    )

    jacobian_to_wavenumber_density = (
        spectral_grid["radian_frequency"][number_of_frequencies - 1]
        / last_resolved_wavenumber
        / 4
        / pi
    )

    starting_energy = directional_integral_last_bin * jacobian_to_wavenumber_density

    bounds = log_bounds_wavenumber(roughness_length, friction_velocity, parameters)

    if bounds[1] < log(last_resolved_wavenumber):
        return 0.0

    if bounds[0] < log(last_resolved_wavenumber):
        bounds[0] = log(last_resolved_wavenumber)

    if bounds[1] <= bounds[0]:
        print("warning")
        print(bounds)
        return 0.0

    log_wavenumbers = linspace(bounds[0], bounds[1], 100)
    wavenumbers = exp(log_wavenumbers)

    saturation_spectrum = saturation_spectrum_parametrization(
        wavenumbers,
        starting_energy,
        last_resolved_wavenumber,
        friction_velocity,
        parameters,
    )
    tail_spectrum = saturation_spectrum * wavenumbers**-3

    windinput = wind_input_tail(
        log_wavenumbers, roughness_length, friction_velocity, tail_spectrum, parameters
    )

    angular_frequency = dispersion(
        wavenumbers,
        parameters["gravitational_acceleration"],
        parameters["surface_tension"],
    )

    stress = 0.0
    for wavenumber_index in range(1, wavenumbers.shape[0]):
        wavenumber_step = (
            wavenumbers[wavenumber_index] - wavenumbers[wavenumber_index - 1]
        )

        stress += (
            angular_frequency[wavenumber_index - 1]
            * windinput[wavenumber_index - 1]
            / 2
            + angular_frequency[wavenumber_index] * windinput[wavenumber_index] / 2
        ) * wavenumber_step

    eastward_stress = (
        stress * parameters["water_density"] * cos(wind_direction_radian),
    )
    northward_stress = (
        stress * parameters["water_density"] * sin(wind_direction_radian),
    )

    return eastward_stress, northward_stress


def wind_stress_tail(
    starting_energy,
    starting_wavenumber,
    roughness_length,
    friction_velocity,
    log_wavenumbers,
    parameters,
):
    wavenumbers = exp(log_wavenumbers)

    saturation_spectrum = saturation_spectrum_parametrization(
        wavenumbers, starting_energy, starting_wavenumber, friction_velocity, parameters
    )

    tail_spectrum = saturation_spectrum * wavenumbers**-3

    windinput = wind_input_tail(
        log_wavenumbers, roughness_length, friction_velocity, tail_spectrum, parameters
    )

    angular_frequency = dispersion(
        wavenumbers,
        parameters["gravitational_acceleration"],
        parameters["surface_tension"],
    )

    stress = empty(wavenumbers.shape[0])
    for wavenumber_index in range(0, wavenumbers.shape[0]):
        stress[wavenumber_index] = (
            angular_frequency[wavenumber_index] * windinput[wavenumber_index]
        )

    return stress * parameters["water_density"]


# ----
# Helper functions Integration domains
# ----
@njit()
def log_bounds_wavenumber(roughness_length, friction_velocity, parameters):
    """
    Find the lower bound of the integration domain for JB2022. Since in this region we have gravity waves we may use
    the approximation from the previous Wam cycle to find this wavenumber. We further assume deep water.

    :param friction_velocity:
    :param effective_charnock:
    :param vonkarman_constant:
    :param wave_age_tuning_parameter:
    :param gravitational_acceleration:
    :return:
    """
    args = (roughness_length, friction_velocity, parameters)

    # find the right root
    log_upper_bound = numba_newton_raphson(
        miles_mu_cutoff,
        log(1 / roughness_length),
        args,
        (-inf, log(1 / roughness_length)),
        verbose=False,
        name="log bound wavenumber 1",
    )

    args_old = (
        roughness_length
        * parameters["gravitational_acceleration"]
        / friction_velocity**2,
        parameters["vonkarman_constant"],
        parameters["wave_age_tuning_parameter"],
    )

    # find the location of the lower boundary of the integration domain. THis is where
    # loglog_mu = 0
    x0 = numba_newton_raphson(
        log_dimensionless_critical_height,
        log(0.01),
        args_old,
        (-5, log_upper_bound),
        verbose=False,
        name="log bound wavenumber 2",
    )
    guess = x0

    while True:
        # print(guess, log_upper_bound)
        log_lower_bound = numba_newton_raphson(
            miles_mu,
            guess,
            args,
            (-10, log_upper_bound),
            verbose=False,
            name="log bound wavenumber 3",
        )
        if log_lower_bound < log_upper_bound * 0.98:
            break
        else:
            guess = guess - 1

    return array([log_lower_bound, log_upper_bound])


# ----
# Helper functions growth parameter
# ----
@njit(cache=True)
def miles_mu(log_wavenumber, roughness_length, friction_velocity, parameters):
    vonkarman_constant = parameters["vonkarman_constant"]
    wave_age_tuning_parameter = parameters["wave_age_tuning_parameter"]
    gravitational_acceleration = parameters["gravitational_acceleration"]
    surface_tension = parameters["surface_tension"]

    wavenumber = exp(log_wavenumber)
    wavespeed = celerity(wavenumber, gravitational_acceleration, surface_tension)

    return (
        log_wavenumber
        + log(roughness_length)
        + vonkarman_constant
        / (friction_velocity / wavespeed + wave_age_tuning_parameter)
    )


@njit(cache=True)
def miles_mu_cutoff(log_wavenumber, roughness_length, friction_velocity, parameters):
    mu = exp(miles_mu(log_wavenumber, roughness_length, friction_velocity, parameters))
    return log(mu - 0.25 * tanh(4 * friction_velocity**4))


@njit(cache=True)
def wind_input_tail(
    log_wavenumbers, roughness_length, friction_velocity, tail_spectrum, parameters
):
    vonkarman_constant = parameters["vonkarman_constant"]
    growth_parameter_betamax = parameters["growth_parameter_betamax"]
    width_factor = parameters["width_factor"]

    number_of_wavenumbers = log_wavenumbers.shape[0]
    windinput = empty(number_of_wavenumbers)

    mu = miles_mu(log_wavenumbers, roughness_length, friction_velocity, parameters)

    epsilon = parameters["air_density"] / parameters["water_density"]
    wavenumber = exp(log_wavenumbers)
    wave_speed = celerity(
        wavenumber, parameters["gravitational_acceleration"], 0.0
    )  # parameters["surface_tension"])
    angular_frequency = dispersion(
        wavenumber,
        parameters["gravitational_acceleration"],
        parameters["surface_tension"],
    )
    k3w = three_wave_starting_wavenumber(
        parameters["gravitational_acceleration"],
        parameters["surface_tension"],
        friction_velocity,
    )

    miles_cutoff = log(1 + 0.25 * tanh(4 * friction_velocity**4))
    for wavenumber_index in range(0, number_of_wavenumbers):
        if mu[wavenumber_index] > miles_cutoff:
            windinput[wavenumber_index] = 0.0
            continue

        if wavenumber[wavenumber_index] > k3w:
            integration_width = 0.35 + width_factor * tanh(3 * friction_velocity**2)
        else:
            integration_width = 0.65

        linear_growth_parameter = (
            angular_frequency[wavenumber_index]
            * mu[wavenumber_index] ** 4
            * exp(mu[wavenumber_index])
            * growth_parameter_betamax
            / vonkarman_constant**2
            * epsilon
            * friction_velocity**2
            / wave_speed[wavenumber_index] ** 2
        )

        N2 = (
            integration_width
            * tail_spectrum[wavenumber_index]
            * linear_growth_parameter
            * wavenumber[wavenumber_index] ** 2
            / vonkarman_constant
            / epsilon
            / friction_velocity
        )
        N1 = N2 / 6.0
        nonlinear_correction = (1.0 + N1) / (1.0 + N2)
        growth_parameter = linear_growth_parameter * nonlinear_correction

        windinput[wavenumber_index] = (
            integration_width * growth_parameter * tail_spectrum[wavenumber_index]
        )

    return windinput


# ----
# Helper functions spectral parametrization tail
# ----
@njit(cache=True)
def saturation_spectrum_parametrization(
    wavenumbers,
    energy_at_starting_wavenumber,
    starting_wavenumber,
    friction_velocity,
    parameters,
):
    """
    Saturation spectrum accordin to the VIERS model (adapted from JB2023)

    :param wavenumbers: set of wavenumbers
    :param energy_at_starting_wavenumber: variance density as a function of wavenumber,
        scaled such that int(e(k) dk = variance. This varies from Peter's work who uses an energy E such that e = E*k
        with k the wavenumber which originates from a transfer to polar coordinates of the 2d wavenumber spectrum.

    :param gravitational_acceleration: gravitational
    :param surface_tension:
    :param friction_velocity:
    :return:
    """

    gravitational_acceleration = parameters["gravitational_acceleration"]
    surface_tension = parameters["surface_tension"]

    # Starting wave number where we switch on 3-wave interactions.
    three_wave_start = three_wave_starting_wavenumber(
        gravitational_acceleration, surface_tension, friction_velocity
    )

    number_of_wavenumbers = wavenumbers.shape[0]
    spectrum = empty(number_of_wavenumbers)

    # Saturation in the "saturation range", we assume a k**-3 spectrum here (f**-5)
    saturation_at_boundary = energy_at_starting_wavenumber * starting_wavenumber**3

    # Strength of the 3-wave interactin parameter. This is directly taken from the VIERS work - as it was not
    # specified what was used in JB23.
    strength_three_wave_interactions = (
        3 * pi / 16 * (tanh(2 * (sqrt(wavenumbers / three_wave_start) - 1)) + 1)
    )

    # Strength at the point where we turn on the interactions
    strength_three_wave_interactions_start = 3 * pi / 16

    energy_flux_at_boundary = (
        strength_three_wave_interactions_start
        * saturation_at_boundary**2
        * celerity(three_wave_start, gravitational_acceleration, surface_tension) ** 4
        / group_velocity(three_wave_start, gravitational_acceleration, surface_tension)
    )

    k0 = sqrt(gravitational_acceleration / surface_tension)
    c0 = (gravitational_acceleration * surface_tension) ** (1 / 4)
    for wavenumber_index in range(number_of_wavenumbers):
        if wavenumbers[wavenumber_index] > three_wave_start:
            scaling_constant = sqrt(
                energy_flux_at_boundary
                / 2
                / strength_three_wave_interactions[wavenumber_index]
            ) * c0 ** (-3 / 2)
            y = wavenumbers[wavenumber_index] / k0
            spectrum[wavenumber_index] = (
                scaling_constant
                * y
                * sqrt(1 + 3 * y**2)
                / ((1 + y**2) * (y + y**3) ** (1 / 4))
            )

        else:
            spectrum[wavenumber_index] = saturation_at_boundary

    return spectrum


@njit(cache=True)
def three_wave_starting_wavenumber(
    gravitational_acceleration, surface_tension, friction_velocity
):
    """
    Starting wavenumber for the capilary-gravity part. See JB2023, eq 41 and 42.
    :param gravitational_acceleration:
    :param surface_tension:
    :param friction_velocity:
    :return:
    """
    return (
        sqrt(gravitational_acceleration / surface_tension)
        * 1
        / (1.48 + 2.05 * friction_velocity)
    )


# ----
# Helper functions grav. cap. waves
# ----
@njit(cache=True)
def dispersion(wavenumber, gravitational_acceleration, surface_tension):
    return sqrt(
        gravitational_acceleration * wavenumber + surface_tension * wavenumber**3
    )


@njit(cache=True)
def celerity(wavenumber, gravitational_acceleration, surface_tension):
    return sqrt(gravitational_acceleration / wavenumber + surface_tension * wavenumber)


@njit(cache=True)
def group_velocity(wavenumber, gravitational_acceleration, surface_tension):
    return (
        1
        / 2
        * (gravitational_acceleration + 3 * surface_tension * wavenumber**2)
        / sqrt(
            gravitational_acceleration * wavenumber + surface_tension * wavenumber**3
        )
    )
