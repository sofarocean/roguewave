from typing import Tuple

from numba import prange, jit
from numba_progress import ProgressBar
from numba.typed import List as NumbaList
from numpy import inf, nan, empty, isnan, zeros
from numpy.typing import NDArray

from roguewave.wavephysics.balance.dissipation import _bulk_dissipation_direction_point
from roguewave.wavephysics.balance._numba_settings import (
    numba_nocache_parallel,
    numba_nocache,
    numba_default,
)
from roguewave.wavephysics.balance.stress import (
    _roughness_estimate_point,
    _total_stress_point,
)
from roguewave.wavephysics.balance.solvers import numba_newton_raphson
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from roguewave.wavephysics.balance.balance import SourceTermBalance
from xarray import DataArray, Dataset
from roguewave import FrequencyDirectionSpectrum


def windspeed_and_direction_from_spectra(
    balance: SourceTermBalance,
    guess_u10: DataArray,
    spectrum: FrequencyDirectionSpectrum,
    jacobian=False,
    jacobian_parameters=None,
    time_derivative_spectrum: FrequencyDirectionSpectrum = None,
    direction_iteration=False,
) -> Dataset:
    """

    :param bulk_rate:
    :param guess_u10:
    :param guess_direction:
    :param spectrum:
    :return:
    """
    disable = spectrum.number_of_spectra < 100
    if time_derivative_spectrum is None:
        time_derivative_spectrum = zeros(spectrum.shape())
    else:
        time_derivative_spectrum = time_derivative_spectrum.variance_density.values

    with ProgressBar(
        total=spectrum.number_of_spectra,
        disable=disable,
        desc=f"Estimating U10 from {balance.generation.name} and {balance.dissipation.name} "
        f"wind and dissipation source terms",
    ) as progress_bar:
        if not jacobian:
            speed, direction = _u10_from_spectra(
                variance_density=spectrum.variance_density.values,
                guess_u10=guess_u10.values,
                depth=spectrum.depth.values,
                wind_source_term_function=balance.generation._wind_source_term_function,
                tail_stress_parametrization_function=balance.generation._tail_stress_parametrization_function,
                dissipation_source_term_function=balance.dissipation._dissipation_function,
                parameters_generation=balance.generation.parameters,
                parameters_dissipation=balance.dissipation.parameters,
                spectral_grid=balance.generation.spectral_grid(spectrum),
                progress_bar=progress_bar,
                time_derivative_spectrum=time_derivative_spectrum,
                direction_iteration=direction_iteration,
            )
        else:
            if jacobian_parameters is None:
                raise ValueError(
                    "If gradients are requested a parameter list is required"
                )

            speed, direction, grad = _u10_from_spectra_gradient(
                variance_density=spectrum.variance_density.values,
                guess_u10=guess_u10.values,
                depth=spectrum.depth.values,
                wind_source_term_function=balance.generation._wind_source_term_function,
                tail_stress_parametrization_function=balance.generation._tail_stress_parametrization_function,
                dissipation_source_term_function=balance.dissipation._dissipation_function,
                grad_parameters=NumbaList(jacobian_parameters),
                parameters_generation=balance.generation.parameters,
                parameters_dissipation=balance.dissipation.parameters,
                spectral_grid=balance.generation.spectral_grid(spectrum),
                time_derivative_spectrum=time_derivative_spectrum,
                progress_bar=progress_bar,
                direction_iteration=direction_iteration,
            )
            grad = DataArray(data=grad)

    u10 = DataArray(
        data=speed,
        dims=spectrum.dims_space_time,
        coords=spectrum.coords_space_time,
    )

    direction = DataArray(
        data=direction,
        dims=spectrum.dims_space_time,
        coords=spectrum.coords_space_time,
    )

    if jacobian:
        return Dataset(data_vars={"u10": u10, "direction": direction, "jacobian": grad})
    else:
        return Dataset(data_vars={"u10": u10, "direction": direction})


@jit(**numba_nocache)
def _u10_from_bulk_rate_point(
    bulk_rate,
    variance_density,
    guess_u10,
    guess_direction,
    depth,
    spectral_grid,
    parameters,
    wind_source_term_function,
    tail_stress_parametrization_function,
    time_derivative_spectrum,
    direction_iteration,
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

    direction = guess_direction
    u10 = guess_u10

    if bulk_rate == 0.0:
        return 0.0, guess_direction

    # We are tryomg to solve for U10; ~ accuracy of 0.01m/s is sufficient. We do not really care about relative accuracy
    # - for low winds (<1m/s) answers are really inaccurate anyway
    atol = 1.0e-2
    rtol = 1.0
    numerical_stepsize = 1e-3

    if direction_iteration:
        niter = 20
    else:
        niter = 1

    for ii in range(0, niter):
        wind_guess = (u10, direction, "u10")
        roughness_memory = [-1.0]
        args = (
            roughness_memory,
            variance_density,
            wind_guess,
            depth,
            wind_source_term_function,
            tail_stress_parametrization_function,
            spectral_grid,
            parameters,
            bulk_rate,
            time_derivative_spectrum,
        )

        try:
            u10 = numba_newton_raphson(
                _u10_iteration_function,
                u10,
                args,
                (0, inf),
                atol=atol,
                rtol=rtol,
                numerical_stepsize=numerical_stepsize,
            )
        except:
            u10 = nan
            break

        if direction_iteration:
            wind_guess = (u10, direction, "u10")
            _, new_direction = _total_stress_point(
                roughness_memory[0],
                variance_density,
                wind_guess,
                depth,
                wind_source_term_function,
                tail_stress_parametrization_function,
                spectral_grid,
                parameters,
            )

            direction_delta = (new_direction - direction + 180) % 360 - 180

            if abs(direction_delta) < 1.0:
                direction = new_direction
                break
            elif abs(direction_delta) < 10.0:
                direction = new_direction
            else:
                # Some under-relaxation
                direction = (direction + 0.5 * direction_delta) % 360

    return u10, direction


@jit(**numba_nocache)
def _u10_iteration_function(
    u10,
    memory_list,
    variance_density,
    wind_guess,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    spectral_grid,
    parameters,
    bulk_rate,
    time_derivative_spectrum,
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
        tail_stress_parametrization_function,
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

    # Calculate the contribution of dEdt integrated over the spectrum. We only include spectral values in region where
    # there is a positive wind input
    bulk_time_derivative_spectrum = spectral_time_derivative_in_active_region(
        time_derivative_spectrum,
        generation,
        spectral_grid,
    )

    # Integrate the input and return the difference of the current guess with the desired bulk rate
    return (
        numba_integrate_spectral_data(generation, spectral_grid)
        - bulk_rate
        - bulk_time_derivative_spectrum
    )


@jit(**numba_default)
def spectral_time_derivative_in_active_region(
    time_derivative_spectrum: NDArray,
    generation: NDArray,
    spectral_grid,
):
    frequency_step = spectral_grid["frequency_step"]
    direction_step = spectral_grid["direction_step"]
    number_of_directions = time_derivative_spectrum.shape[1]
    number_of_frequencies = time_derivative_spectrum.shape[0]

    bulk_time_derivative_spectrum = 0.0

    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            if generation[frequency_index, direction_index] > 0.0:
                bulk_time_derivative_spectrum += (
                    time_derivative_spectrum[frequency_index, direction_index]
                    * direction_step[direction_index]
                    * frequency_step[frequency_index]
                )
    return bulk_time_derivative_spectrum


@jit(**numba_nocache)
def _u10_from_spectra_point(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    dissipation_source_term_function,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    time_derivative_spectrum,
    direction_iteration,
) -> Tuple[float, float]:
    """

    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param depth:
    :param parameters:
    :param spectral_grid:
    :param progress_bar:
    :return:
    """

    direction, bulk_rate = _bulk_dissipation_direction_point(
        variance_density,
        depth,
        dissipation_source_term_function,
        spectral_grid,
        parameters_dissipation,
    )

    # Note dissipation is negatve- but our target bulk wind generation is positive
    u10, direction = _u10_from_bulk_rate_point(
        -bulk_rate,
        variance_density,
        guess_u10,
        direction,
        depth,
        spectral_grid,
        parameters_generation,
        wind_source_term_function,
        tail_stress_parametrization_function,
        time_derivative_spectrum,
        direction_iteration,
    )
    return u10, direction


# ----------------------------------------------------------------------------------------------------------------------
# Functions for training (gradients with regard to parameter)
# ----------------------------------------------------------------------------------------------------------------------


@jit(**numba_nocache)
def _u10_parameter_gradient(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    dissipation_source_term_function,
    grad_parameters,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    time_derivative_spectrum,
    direction_iteration,
) -> (float, float, NDArray):
    """
    Function to numerically calculate gradients for the requested coeficients
    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param depth:
    :param parameters:
    :param spectral_grid:
    :param progress_bar:
    :return:
    """

    # Calculate the zero point
    direction, bulk_rate = _bulk_dissipation_direction_point(
        variance_density,
        depth,
        dissipation_source_term_function,
        spectral_grid,
        parameters_dissipation,
    )

    # Note dissipation is negatve- but our target bulk wind generation is positive
    u10, direction = _u10_from_bulk_rate_point(
        -bulk_rate,
        variance_density,
        guess_u10,
        direction,
        depth,
        spectral_grid,
        parameters_generation,
        wind_source_term_function,
        tail_stress_parametrization_function,
        time_derivative_spectrum,
        direction_iteration,
    )
    grad = empty(len(grad_parameters))
    if isnan(u10):
        grad[:] = 0
        return u10, direction, grad

    for index, param in enumerate(grad_parameters):
        perturbed_parameters_dissipation = parameters_dissipation.copy()
        perturbed_parameters_generation = parameters_generation.copy()

        if param in parameters_dissipation:
            step = 0.05 * abs(perturbed_parameters_dissipation[param])
            perturbed_parameters_dissipation[param] += step

            dissipation = dissipation_source_term_function(
                variance_density=variance_density,
                depth=depth,
                spectral_grid=spectral_grid,
                parameters=perturbed_parameters_dissipation,
            )

            new_bulk_rate = numba_integrate_spectral_data(dissipation, spectral_grid)

        else:
            step = 0.05 * abs(perturbed_parameters_generation[param])
            perturbed_parameters_generation[param] += step
            new_bulk_rate = bulk_rate

        new_u10, direction = _u10_from_bulk_rate_point(
            -new_bulk_rate,
            variance_density,
            u10,
            direction,
            depth,
            spectral_grid,
            perturbed_parameters_generation,
            wind_source_term_function,
            tail_stress_parametrization_function,
            time_derivative_spectrum,
            direction_iteration,
        )
        if isnan(new_u10):
            grad[index] = 0
        else:
            grad[index] = (new_u10 - u10) / step

    return u10, direction, grad


# ----------------------------------------------------------------------------------------------------------------------
# Apply to all spatial points
# ----------------------------------------------------------------------------------------------------------------------


@jit(**numba_nocache_parallel)
def _u10_from_spectra_gradient(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    tail_stress_parametrization_function,
    dissipation_source_term_function,
    grad_parameters,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    time_derivative_spectrum,
    progress_bar: ProgressBar = None,
    direction_iteration=False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """

    :param variance_density:
    :param guess_u10:
    :param depth:
    :param wind_source_term_function:
    :param dissipation_source_term_function:
    :param parameters_generation:
    :param parameters_dissipation:
    :param spectral_grid:
    :param progress_bar:
    :return:
    """
    number_of_points = variance_density.shape[0]
    u10 = empty((number_of_points))
    direction = empty((number_of_points))
    grad = empty((number_of_points, len(grad_parameters)))
    for point_index in prange(number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        (
            u10[point_index],
            direction[point_index],
            grad[point_index, :],
        ) = _u10_parameter_gradient(
            variance_density[point_index, :, :],
            guess_u10[point_index],
            depth[point_index],
            wind_source_term_function,
            tail_stress_parametrization_function,
            dissipation_source_term_function,
            grad_parameters,
            parameters_generation,
            parameters_dissipation,
            spectral_grid,
            time_derivative_spectrum[point_index, :, :],
            direction_iteration,
        )
    return u10, direction, grad


@jit(**numba_nocache_parallel)
def _u10_from_spectra(
    variance_density: NDArray,
    guess_u10: NDArray,
    depth: NDArray,
    wind_source_term_function,
    tail_stress_parametrization_function,
    dissipation_source_term_function,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    progress_bar: ProgressBar = None,
    time_derivative_spectrum=None,
    direction_iteration=False,
) -> Tuple[NDArray, NDArray]:
    """

    :param variance_density:
    :param guess_u10:
    :param depth:
    :param wind_source_term_function:
    :param dissipation_source_term_function:
    :param parameters_generation:
    :param parameters_dissipation:
    :param spectral_grid:
    :param progress_bar:
    :return:
    """
    number_of_points = variance_density.shape[0]
    u10 = empty((number_of_points))
    direction = empty((number_of_points))

    for point_index in prange(number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        u10[point_index], direction[point_index] = _u10_from_spectra_point(
            variance_density[point_index, :, :],
            guess_u10[point_index],
            depth[point_index],
            wind_source_term_function,
            tail_stress_parametrization_function,
            dissipation_source_term_function,
            parameters_generation,
            parameters_dissipation,
            spectral_grid,
            time_derivative_spectrum[point_index, :, :],
            direction_iteration,
        )
    return u10, direction
