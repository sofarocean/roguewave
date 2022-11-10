from roguewave import FrequencyDirectionSpectrum
from xarray import DataArray
from numpy import pi, cos, sin, arctan2, empty
from numpy.typing import NDArray
from typing import Literal, Tuple
from roguewave.wavephysics.balance.source_term import SourceTerm
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from roguewave.wavetheory.lineardispersion import inverse_intrinsic_dispersion_relation
from numba import njit, prange

breaking_parametrization = Literal["st6", "st4"]


class Dissipation(SourceTerm):
    name = "Dissipation"

    def __init__(self, parameters):
        super(Dissipation, self).__init__(parameters)
        self._dissipation_function = None

    def rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:
        data = _dissipation(
            spectrum.variance_density.values,
            spectrum.depth.values,
            self._dissipation_function,
            self.spectral_grid(spectrum),
            self.parameters,
        )
        return DataArray(data=data, coords=spectrum.coords(), dims=spectrum.dims)

    def bulk_rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:

        data = _bulk_dissipation(
            spectrum.variance_density.values,
            spectrum.depth.values,
            self._dissipation_function,
            self.spectral_grid(spectrum),
            self.parameters,
        )
        return DataArray(
            data=data, coords=spectrum.coords_space_time, dims=spectrum.dims_space_time
        )

    def mean_direction_degrees(self, spectrum: FrequencyDirectionSpectrum):
        data, _ = _bulk_dissipation_direction(
            spectrum.variance_density.values,
            spectrum.depth.values,
            self._dissipation_function,
            self.spectral_grid(spectrum),
            self.parameters,
        )
        return DataArray(
            data=data, coords=spectrum.coords_space_time, dims=spectrum.dims_space_time
        )


@njit(cache=False, parallel=False)
def _dissipation(
    variance_density, depth, dissipation_source_term_function, spectral_grid, parameters
) -> NDArray:
    (
        number_of_points,
        number_of_frequencies,
        number_of_directions,
    ) = variance_density.shape

    dissipation = empty((number_of_points, number_of_frequencies, number_of_directions))
    for point_index in prange(number_of_points):
        diss = dissipation_source_term_function(
            variance_density[point_index, :, :],
            depth[point_index],
            spectral_grid,
            parameters,
        )
        dissipation[point_index, :, :] = diss
    return dissipation


@njit(cache=False, parallel=False)
def _bulk_dissipation(
    variance_density, depth, dissipation_source_term_function, spectral_grid, parameters
) -> NDArray:
    number_of_points = variance_density.shape[0]
    dissipation = empty((number_of_points))

    for point_index in prange(number_of_points):
        diss = dissipation_source_term_function(
            variance_density=variance_density[point_index, :, :],
            depth=depth[point_index],
            spectral_grid=spectral_grid,
            parameters=parameters,
        )
        dissipation[point_index] = numba_integrate_spectral_data(diss, spectral_grid)
    return dissipation


@njit(cache=False, parallel=False)
def _bulk_dissipation_direction(
    variance_density, depth, dissipation_source_term_function, spectral_grid, parameters
) -> Tuple[NDArray, NDArray]:
    number_of_points = variance_density.shape[0]
    direction = empty((number_of_points))
    bulk = empty((number_of_points))

    for point_index in prange(number_of_points):
        direction[point_index], bulk[point_index] = _bulk_dissipation_direction_point(
            variance_density[point_index, :, :],
            depth[point_index],
            dissipation_source_term_function,
            spectral_grid,
            parameters,
        )

    return direction, bulk


@njit(cache=True)
def _bulk_dissipation_direction_point(
    variance_density, depth, dissipation_source_term_function, spectral_grid, parameters
) -> Tuple[float, float]:
    number_of_frequency = variance_density.shape[0]
    number_of_direction = variance_density.shape[1]

    dissipation = dissipation_source_term_function(
        variance_density=variance_density,
        depth=depth,
        spectral_grid=spectral_grid,
        parameters=parameters,
    )

    bulk = numba_integrate_spectral_data(dissipation, spectral_grid)
    radian_frequency = spectral_grid["radian_frequency"]
    radian_direction = spectral_grid["radian_direction"]
    frequency_step = spectral_grid["frequency_step"]
    direction_step = spectral_grid["direction_step"]
    wave_number = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    cosine = cos(radian_direction)
    sine = sin(radian_direction)

    kx = 0.0
    ky = 0.0

    # Disspation weighted average wave number to guestimate the wind direction. Note dissipation is negative- hence the
    # minus signs.
    for frequency_index in range(number_of_frequency):
        for direction_index in range(number_of_direction):
            kx -= (
                wave_number[frequency_index]
                * cosine[direction_index]
                * dissipation[frequency_index, direction_index]
                * frequency_step[frequency_index]
                * direction_step[direction_index]
            )

            ky -= (
                wave_number[frequency_index]
                * sine[direction_index]
                * dissipation[frequency_index, direction_index]
                * frequency_step[frequency_index]
                * direction_step[direction_index]
            )

    return (arctan2(ky, kx) * 180 / pi) % 360.0, bulk
