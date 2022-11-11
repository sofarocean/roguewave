from roguewave import (
    FrequencyDirectionSpectrum,
)
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from roguewave.wavephysics.balance.source_term import SourceTerm
from roguewave.wavephysics.balance.solvers import (
    numba_newton_raphson,
    numba_fixed_point_iteration,
)
from xarray import DataArray, zeros_like
from typing import Literal, Tuple
from numpy import inf, empty, isnan, nan, arctan2, sqrt, pi, cos, sin, exp, any, log
from numba import njit, types, prange
from numba.typed import Dict as NumbaDict
from numpy.typing import NDArray
from roguewave.wavetheory import inverse_intrinsic_dispersion_relation

TWindInputType = Literal["u10", "friction_velocity", "ustar"]


class WindGeneration(SourceTerm):
    name = "base"

    def __init__(self, parmaters):
        super(WindGeneration, self).__init__(parmaters)
        self._wind_source_term_function = None

    def rate(
        self,
        spectrum: FrequencyDirectionSpectrum,
        speed: DataArray,
        direction: DataArray,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:

        wind = (speed.values, direction.values, wind_speed_input_type)
        wind_input = _wind_generation(
            spectrum.variance_density.values,
            wind=wind,
            depth=spectrum.depth.values,
            roughness_length=roughness_length.values,
            wind_source_term_function=self._wind_source_term_function,
            spectral_grid=self.spectral_grid(spectrum),
            parameters=self.parameters,
        )
        return DataArray(data=wind_input, dims=spectrum.dims, coords=spectrum.coords())

    def bulk_rate(
        self,
        spectrum: FrequencyDirectionSpectrum,
        speed: DataArray,
        direction: DataArray,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        wind = (speed.values, direction.values, wind_speed_input_type)
        wind_input = _bulk_wind_generation(
            spectrum.variance_density.values,
            wind,
            spectrum.depth.values,
            roughness_length.values,
            self._wind_source_term_function,
            self.spectral_grid(spectrum),
            self.parameters,
        )
        return DataArray(
            data=wind_input,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

    def roughness(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length_guess: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        if roughness_length_guess is None:
            roughness_length_guess = zeros_like(speed) - 1

        wind = (speed.values, direction.values, wind_speed_input_type)

        rougness = _roughness_estimate(
            guess=roughness_length_guess.values,
            variance_density=spectrum.variance_density.values,
            wind=wind,
            depth=spectrum.depth.values,
            wind_source_term_function=self._wind_source_term_function,
            spectral_grid=self.spectral_grid(spectrum),
            parameters=self.parameters,
        )
        return DataArray(
            data=rougness,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )


@njit(cache=True, fastmath=True)
def _wave_supported_stress_point(
    wind_input: NDArray, depth, spectral_grid, parameters
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

    common_factor = (
        parameters["gravitational_acceleration"] * parameters["water_density"]
    )
    if depth == inf:
        wavenumber = radian_frequency**2 / gravitational_acceleration
    else:
        wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)

    cosine_step = cos(radian_direction) * direction_step
    sine_step = sin(radian_direction) * direction_step

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

    wave_stress_magnitude = sqrt(stress_north**2 + stress_east**2) * common_factor
    wave_stress_direction = (arctan2(stress_north, stress_east) * 180 / pi) % 360

    return wave_stress_magnitude, wave_stress_direction


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


@njit()
def _roughness_estimate_point(
    guess,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
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
        spectral_grid,
        parameters,
        generation,
    )
    return numba_fixed_point_iteration(
        _roughness_iteration_function, guess, function_arguments, bounds=(0, inf)
    )


# ----------------------------------------------------------------------------------------------------------------------
# Apply to all spatial points
# ----------------------------------------------------------------------------------------------------------------------
@njit()
def _u10_from_bulk_rate_point(
    bulk_rate,
    variance_density,
    guess_u10,
    guess_direction,
    depth,
    spectral_grid,
    parameters,
    wind_source_term_function,
):
    """

    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param guess_direction:
    :param depth:
    :param spectral_grid:
    :param parameters:
    :return:
    """
    wind_guess = (guess_u10, guess_direction, "u10")
    roughness_memory = [-1.0]
    args = (
        roughness_memory,
        variance_density,
        wind_guess,
        depth,
        wind_source_term_function,
        spectral_grid,
        parameters,
        bulk_rate,
    )
    if bulk_rate == 0.0:
        return 0.0

    # We are tryomg to solve for U10; ~ accuracy of 0.01m/s is sufficient. We do not really care about relative accuracy
    # - for low winds (<1m/s) answers are really inaccurate anyway
    atol = 1.0e-2
    rtol = 1.0
    numerical_stepsize = 1e-3
    try:
        u10 = numba_newton_raphson(
            _u10_iteration_function,
            guess_u10,
            args,
            (0, inf),
            atol=atol,
            rtol=rtol,
            numerical_stepsize=numerical_stepsize,
        )
    except:
        u10 = nan
    return u10


@njit()
def _wind_generation(
    variance_density,
    wind,
    depth,
    roughness_length,
    wind_source_term_function,
    spectral_grid,
    parameters,
) -> NDArray:
    (
        number_of_points,
        number_of_frequencies,
        number_of_directions,
    ) = variance_density.shape

    generation = empty((number_of_points, number_of_frequencies, number_of_directions))
    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        wind_generation = wind_source_term_function(
            variance_density[point_index, :, :],
            wind_at_point,
            depth[point_index],
            roughness_length[point_index],
            spectral_grid,
            parameters,
        )
        generation[point_index, :, :] = wind_generation
    return generation


@njit(cache=True)
def _bulk_wind_generation(
    variance_density,
    wind,
    depth,
    roughness_length,
    wind_source_term_function,
    spectral_grid,
    parameters,
) -> NDArray:
    number_of_points = variance_density.shape[0]
    generation = empty((number_of_points))

    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        wind_generation = wind_source_term_function(
            variance_density=variance_density[point_index, :, :],
            wind=wind_at_point,
            depth=depth[point_index],
            roughness_length=roughness_length[point_index],
            spectral_grid=spectral_grid,
            parameters=parameters,
        )
        generation[point_index] = numba_integrate_spectral_data(
            wind_generation, spectral_grid
        )
    return generation


@njit()
def _roughness_estimate(
    guess,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
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
                spectral_grid=spectral_grid,
                parameters=parameters,
            )

    return roughness


@njit(cache=True)
def _wave_supported_stress(
    variance_density,
    wind,
    depth,
    roughness_length,
    wind_source_term_function,
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
        wind_generation = wind_source_term_function(
            variance_density=variance_density[point_index, :, :],
            wind=wind_at_point,
            depth=depth[point_index],
            roughness_length=roughness_length[point_index],
            spectral_grid=spectral_grid,
            parameters=parameters,
        )

        (
            magnitude[point_index],
            direction[point_index],
        ) = _wave_supported_stress_point(
            wind_generation, depth[point_index], spectral_grid, parameters
        )
    return magnitude, direction


@njit(parallel=False)
def _u10_from_bulk_rate(
    bulk_rate,
    variance_density,
    guess_u10,
    guess_direction,
    depth,
    wind_source_term_function,
    parameters,
    spectral_grid,
    progress_bar=None,
) -> NDArray:
    """

    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param guess_direction:
    :param depth:
    :param parameters:
    :param spectral_grid:
    :param progress_bar:
    :return:
    """
    number_of_points = variance_density.shape[0]
    u10 = empty((number_of_points))
    for point_index in prange(number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        u10[point_index] = _u10_from_bulk_rate_point(
            bulk_rate[point_index],
            variance_density[point_index, :, :],
            guess_u10[point_index],
            guess_direction[point_index],
            depth[point_index],
            spectral_grid=spectral_grid,
            parameters=parameters,
            wind_source_term_function=wind_source_term_function,
        )
    return u10


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------
@njit()
def _roughness_iteration_function(
    roughness_length,
    variance_density,
    wind,
    depth,
    wind_source_term_function,
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
    wave_supported_stress, _ = _wave_supported_stress_point(
        work_array, depth, spectral_grid, parameters
    )

    # Given the stress ratio, correct the Charnock roughness for the presence of waves.
    total_stress = parameters["air_density"] * friction_velocity**2
    stress_ratio = wave_supported_stress / total_stress

    if stress_ratio > 0.9:
        stress_ratio = 0.9

    charnock_roughness = _charnock_relation_point(friction_velocity, parameters)
    return charnock_roughness / sqrt(1 - stress_ratio)


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------
@njit(fastmath=True)
def _u10_iteration_function(
    u10,
    memory_list,
    variance_density,
    wind_guess,
    depth,
    wind_source_term_function,
    spectral_grid,
    parameters,
    bulk_rate,
):
    """
    To find the 10 meter wind that generates a certain bulk input we need to solve the inverse function for the
    wind input term. We define a function

            F(U10) = wind_bulk_rate(u10) - specified_bulk_rate

    and use a zero finder to solve for

            F(U10) = 0.

    This function defines the function F. The zero finder is responsible for calling it with the correct trailing
    arguments.

    :param wind_forcing: Wind input U10 we want to solve for.
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind_guess:
    :param depth:
    :param spectral_grid:
    :param parameters:
    :param bulk_rate:
    :return:
    """
    if u10 == 0.0:
        return -bulk_rate

    wind = (u10, wind_guess[1], wind_guess[2])

    # Estimate the rougness length
    roughness_length = _roughness_estimate_point(
        memory_list[0],
        variance_density,
        wind,
        depth,
        wind_source_term_function,
        spectral_grid,
        parameters,
    )
    memory_list[0] = roughness_length

    # Calculate the wind input source term values
    generation = wind_source_term_function(
        variance_density,
        wind,
        depth,
        roughness_length,
        spectral_grid,
        parameters,
    )

    # Integrate the input and return the difference of the current guess with the desired bulk rate
    return numba_integrate_spectral_data(generation, spectral_grid) - bulk_rate


@njit(fastmath=True)
def _spectral_grid(radian_frequency, radian_direction, frequency_step, direction_step):
    return {
        "radian_frequency": radian_frequency,
        "radian_direction": radian_direction,
        "frequency_step": frequency_step,
        "direction_step": direction_step,
    }


def _numba_parameters(**kwargs):
    _dict = NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)
    for key in kwargs:
        _dict[key] = kwargs[key]

    return _dict
