from roguewave.wavephysics.balance import WindGeneration, Dissipation
from typing import Mapping, Tuple
from roguewave import FrequencyDirectionSpectrum
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from roguewave.wavephysics.balance.generation import _u10_from_bulk_rate_point
from roguewave.wavephysics.balance.dissipation import _bulk_dissipation_direction_point
from xarray import DataArray, Dataset
from numba_progress import ProgressBar
from numba.typed import List as NumbaList
from numba import njit, prange
from numpy.typing import NDArray
from numpy import empty, isnan


class SourceTermBalance:
    def __init__(self, generation: WindGeneration, disspipation: Dissipation):
        self.generation = generation
        self.dissipation = disspipation

    def evaluate_imbalance(
        self,
        wind_speed: DataArray,
        wind_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        return self.generation.rate(
            spectrum,
            wind_speed,
            wind_direction,
        ) + self.dissipation.rate(spectrum)

    def evaluate_bulk_imbalance(
        self,
        wind_speed: DataArray,
        wind_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        return self.generation.bulk_rate(
            spectrum, wind_speed, wind_direction
        ) + self.dissipation.bulk_rate(spectrum)

    def update_parameters(self, parameters: Mapping):
        for key in parameters:
            if key in self.generation._parameters:
                self.generation._parameters[key] = parameters[key]
            elif key in self.dissipation._parameters:
                self.dissipation._parameters[key] = parameters[key]

    def windspeed_and_direction_from_spectra(
        self,
        guess_u10: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        jacobian=False,
        jacobian_parameters=None,
    ) -> Dataset:
        """

        :param bulk_rate:
        :param guess_u10:
        :param guess_direction:
        :param spectrum:
        :return:
        """
        disable = spectrum.number_of_spectra < 100
        with ProgressBar(
            total=spectrum.number_of_spectra,
            disable=disable,
            desc=f"Estimating U10 from {self.generation.name} and {self.dissipation.name} wind and dissipation source terms",
        ) as progress_bar:
            if not jacobian:
                speed, direction = _u10_from_spectra(
                    variance_density=spectrum.variance_density.values,
                    guess_u10=guess_u10.values,
                    depth=spectrum.depth.values,
                    wind_source_term_function=self.generation._wind_source_term_function,
                    dissipation_source_term_function=self.dissipation._dissipation_function,
                    parameters_generation=self.generation.parameters,
                    parameters_dissipation=self.dissipation.parameters,
                    spectral_grid=self.generation.spectral_grid(spectrum),
                    progress_bar=progress_bar,
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
                    wind_source_term_function=self.generation._wind_source_term_function,
                    dissipation_source_term_function=self.dissipation._dissipation_function,
                    grad_parameters=NumbaList(jacobian_parameters),
                    parameters_generation=self.generation.parameters,
                    parameters_dissipation=self.dissipation.parameters,
                    spectral_grid=self.generation.spectral_grid(spectrum),
                    progress_bar=progress_bar,
                )
                grad = DataArray(data=grad)

        u10 = DataArray(
            data=speed,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

        direction = DataArray(
            data=speed,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

        if jacobian:
            return Dataset(
                data_vars={"u10": u10, "direction": direction, "jacobian": grad}
            )
        else:
            return Dataset(data_vars={"u10": u10, "direction": direction})


# ----------------------------------------------------------------------------------------------------------------------
# Apply to all spatial points
# ----------------------------------------------------------------------------------------------------------------------


@njit(parallel=True, nogil=True)
def _u10_from_spectra(
    variance_density: NDArray,
    guess_u10: NDArray,
    depth: NDArray,
    wind_source_term_function,
    dissipation_source_term_function,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    progress_bar: ProgressBar = None,
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
            dissipation_source_term_function,
            parameters_generation,
            parameters_dissipation,
            spectral_grid,
        )
    return u10, direction


@njit(parallel=False)
def _u10_from_spectra_point(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    dissipation_source_term_function,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
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
    u10 = _u10_from_bulk_rate_point(
        -bulk_rate,
        variance_density,
        guess_u10,
        direction,
        depth,
        spectral_grid,
        parameters_generation,
        wind_source_term_function,
    )
    return u10, direction


@njit(parallel=False, cache=True)
def _u10_parameter_gradient(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    dissipation_source_term_function,
    grad_parameters,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
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
    u10 = _u10_from_bulk_rate_point(
        -bulk_rate,
        variance_density,
        guess_u10,
        direction,
        depth,
        spectral_grid,
        parameters_generation,
        wind_source_term_function,
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

        new_u10 = _u10_from_bulk_rate_point(
            -new_bulk_rate,
            variance_density,
            u10,
            direction,
            depth,
            spectral_grid,
            perturbed_parameters_generation,
            wind_source_term_function,
        )
        if isnan(new_u10):
            grad[index] = 0
        else:
            grad[index] = (new_u10 - u10) / step

    return u10, direction, grad


@njit(parallel=True)
def _u10_from_spectra_gradient(
    variance_density,
    guess_u10,
    depth,
    wind_source_term_function,
    dissipation_source_term_function,
    grad_parameters,
    parameters_generation,
    parameters_dissipation,
    spectral_grid,
    progress_bar: ProgressBar = None,
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
            dissipation_source_term_function,
            grad_parameters,
            parameters_generation,
            parameters_dissipation,
            spectral_grid,
        )
    return u10, direction, grad
