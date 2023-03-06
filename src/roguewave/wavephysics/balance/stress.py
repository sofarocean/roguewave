from typing import Tuple

from numba import njit
from numpy import inf, cos, sin, sqrt, arctan2, pi, exp, isnan, nan, empty, log, any
from numpy.typing import NDArray

from roguewave.wavephysics.balance.wam_tail_stress import _charnock_relation_point
from roguewave.wavephysics.balance.solvers import numba_newton_raphson
from roguewave.wavetheory import inverse_intrinsic_dispersion_relation


@njit(fastmath=True)
def _wave_supported_stress_point(
    wind_input: NDArray,
    depth,
    spectral_grid,
    variance_density,
    wind,
    roughness_length,
    tail_stress_parametrization_function,
    parameters,
) -> Tuple[NDArray, NDArray]:
    """
    :param wind_input:
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """

    number_of_frequencies, number_of_directions = wind_input.shape

    radian_frequency = spectral_grid["radian_frequency"]
    radian_direction = spectral_grid["radian_direction"]
    frequency_step = spectral_grid["frequency_step"]
    direction_step = spectral_grid["direction_step"]
    gravitational_acceleration = parameters["gravitational_acceleration"]

    if depth == inf:
        wavenumber = radian_frequency**2 / gravitational_acceleration
    else:
        wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)

    cosine_step = cos(radian_direction) * direction_step
    sine_step = sin(radian_direction) * direction_step

    # Calculate the resolved part of the wavestress over the given input spectrum
    stress_east = 0.0
    stress_north = 0.0
    for frequency_index in range(number_of_frequencies):
        inverse_wave_speed = (
            wavenumber[frequency_index] / radian_frequency[frequency_index]
        ) * frequency_step[frequency_index]

        for direction_index in range(number_of_directions):
            stress_east += (
                cosine_step[direction_index]
                * wind_input[frequency_index, direction_index]
                * inverse_wave_speed
            )

            stress_north += (
                sine_step[direction_index]
                * wind_input[frequency_index, direction_index]
                * inverse_wave_speed
            )

    # calculate the magnitude
    resolved_wave_stress_magnitude = (
        sqrt(stress_north**2 + stress_east**2)
        * parameters["gravitational_acceleration"]
        * parameters["water_density"]
    )

    wave_stress_direction = (arctan2(stress_north, stress_east) * 180 / pi) % 360

    # Add stress contribution due to unresolved waves in the spectral tail (above the last resolved frequency)
    tail_wave_stress_magnitude = tail_stress_parametrization_function(
        variance_density, wind, depth, roughness_length, spectral_grid, parameters
    )

    # Total stress is the sum or resolved and unresolved part.
    wave_stress_magnitude = resolved_wave_stress_magnitude + tail_wave_stress_magnitude
    return wave_stress_magnitude, wave_stress_direction


@njit()
def _roughness_estimate_point(
    guess,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
):
    """

    :param guess:
    :param variance_density:
    :param wind:
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """

    vonkarman_constant = parameters["vonkarman_constant"]
    if guess < 0:
        if wind[2] == "u10":
            drag_coeficient_wu = (0.8 + 0.065 * wind[0]) / 1000
            guess = 10 / exp(vonkarman_constant / sqrt(drag_coeficient_wu))

        elif wind[2] in ["ustar", "friction_velocity"]:
            guess = _charnock_relation_point(wind[0], parameters)
        else:
            raise ValueError("unknown wind forcing")

    if any(isnan(variance_density)):
        return nan

    if wind[0] == 0.0:
        return nan

    generation = empty((variance_density.shape))

    function_arguments = (
        variance_density,
        wind,
        depth,
        wind_source_term_function,
        tail_stress_parametrization_function,
        spectral_grid,
        parameters,
        generation,
    )

    guess = log(guess)

    log_root = numba_newton_raphson(
        _stress_iteration_function,
        guess,
        function_arguments,
        hard_bounds=(-20, 0),
        relative_stepsize=False,
        atol=1e-6,
        rtol=1e-6,
        error_on_max_iter=True,
        max_iterations=100,
        verbose=False,
        name="stress iteration",
        aitken_acceleration=False,
    )
    return exp(log_root)


@njit()
def _roughness_estimate(
    guess,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
) -> NDArray:
    """
    :param guess:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """
    number_of_points = variance_density.shape[0]
    roughness = empty((number_of_points))

    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])

        if isnan(wind[0][point_index]):
            roughness[point_index] = nan
        else:
            roughness[point_index] = _roughness_estimate_point(
                guess=guess[point_index],
                variance_density=variance_density[point_index, :, :],
                wind=wind_at_point,
                depth=depth[point_index],
                wind_source_term_function=wind_source_term_function,
                tail_stress_parametrization_function=tail_stress_parametrization_function,
                spectral_grid=spectral_grid,
                parameters=parameters,
            )

    return roughness


@njit()
def _wave_supported_stress(
    variance_density,
    wind,
    depth,
    roughness_length,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
) -> Tuple[NDArray, NDArray]:
    """
    Calculate the wave supported wind stress.

    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param roughness_length: Surface roughness length in meter.
    :param spectral_grid:
    :param parameters:
    :return:
    """
    number_of_points = variance_density.shape[0]
    magnitude = empty((number_of_points))
    direction = empty((number_of_points))
    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        (magnitude[point_index], direction[point_index],) = _total_stress_point(
            roughness_length[point_index],
            variance_density[point_index, :, :],
            wind=wind_at_point,
            depth=depth[point_index],
            wind_source_term_function=wind_source_term_function,
            tail_stress_parametrization_function=tail_stress_parametrization_function,
            spectral_grid=spectral_grid,
            parameters=parameters,
        )
    return magnitude, direction


@njit()
def _stress_iteration_function(
    log_roughness_length,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
    work_array,
):
    """
    The surface roughness is defined implicitly. We use a fixed-point iteration step to solve for the roughness. This
    is the iteration function used in the fixed point iteration.

    :param roughness_length: Surface roughness length in meter.
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """
    roughness_length = exp(log_roughness_length)

    vonkarman_constant = parameters["vonkarman_constant"]
    elevation = parameters["elevation"]
    if wind[2] == "u10":
        friction_velocity = (
            wind[0] * vonkarman_constant / log(elevation / roughness_length)
        )
    else:
        friction_velocity = wind[0]

    # Get the wind input source term values
    work_array = wind_source_term_function(
        variance_density,
        wind,
        depth,
        roughness_length,
        spectral_grid,
        parameters,
        work_array,
    )

    total_stress_estimate, total_stress_direction_estimate = _total_stress_point(
        roughness_length,
        variance_density,
        wind,
        depth,
        wind_source_term_function,
        tail_stress_parametrization_function,
        spectral_grid,
        parameters,
        work_array,
    )

    return parameters["air_density"] * friction_velocity**2 - total_stress_estimate


@njit()
def _total_stress_point(
    roughness_length,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
    work_array=None,
):
    """

    :param roughness_length: Surface roughness length in meter.
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """

    vonkarman_constant = parameters["vonkarman_constant"]
    elevation = parameters["elevation"]
    if wind[2] == "u10":
        friction_velocity = (
            wind[0] * vonkarman_constant / log(elevation / roughness_length)
        )
    else:
        friction_velocity = wind[0]

    # Get the wind input source term values
    work_array = wind_source_term_function(
        variance_density,
        wind,
        depth,
        roughness_length,
        spectral_grid,
        parameters,
        work_array,
    )

    # Calculate the stress
    wave_supported_stress, stress_direction = _wave_supported_stress_point(
        work_array,
        depth,
        spectral_grid,
        variance_density,
        wind,
        roughness_length,
        tail_stress_parametrization_function,
        parameters,
    )

    viscous_stress = (
        parameters["viscous_stress_parameter"]
        * parameters["air_density"]
        * friction_velocity
        * parameters["air_viscosity"]
        / vonkarman_constant
        / roughness_length
    )

    return wave_supported_stress + viscous_stress, stress_direction
