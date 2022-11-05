from roguewave import FrequencyDirectionSpectrum
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from xarray import DataArray
from roguewave.wavephysics.dissipation.base_class import Dissipation
from numpy.typing import NDArray
from numpy import pi, abs, cos, sin, sqrt, empty, max
from numba import njit
from roguewave.wavetheory import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_velocity,
)


class ST4WaveBreaking(Dissipation):
    def __init__(
        self,
        saturation_breaking_constant=3.501408748021698e-05,
        saturation_breaking_directional_control=0,
        saturation_cosine_power=2,
        saturation_integration_width_degrees=80,
        saturation_threshold=0.0009,
        cumulative_breaking_constant=0.4,
        cumulative_breaking_max_relative_frequency=0.5,
    ):
        self.saturation_breaking_constant = saturation_breaking_constant
        self.saturation_breaking_directional_control = (
            saturation_breaking_directional_control
        )
        self.saturation_cosine_power = saturation_cosine_power
        self.saturation_integration_width_degrees = saturation_integration_width_degrees
        self.saturation_threshold = saturation_threshold
        self.cumulative_breaking_constant = cumulative_breaking_constant
        self.cumulative_breaking_max_relative_frequency = (
            cumulative_breaking_max_relative_frequency
        )

    def bulk_rate(
        self, spectrum: FrequencyDirectionSpectrum, memoize=None
    ) -> DataArray:
        dissipation = st4_bulk_dissipation(
            variance_density=spectrum.variance_density.values,
            depth=spectrum.depth.values,
            radian_frequency=spectrum.radian_frequency.values,
            radian_direction=spectrum.radian_direction.values,
            direction_step=spectrum.direction_step.values,
            frequency_step=spectrum.frequency_step.values,
            saturation_breaking_constant=self.saturation_breaking_constant,
            saturation_breaking_directional_control=self.saturation_breaking_directional_control,
            saturation_cosine_power=self.saturation_cosine_power,
            saturation_integration_width_degrees=self.saturation_integration_width_degrees,
            saturation_threshold=self.saturation_threshold,
            cumulative_breaking_constant=self.cumulative_breaking_constant,
            cumulative_breaking_max_relative_frequency=self.cumulative_breaking_max_relative_frequency,
            number_of_frequencies=spectrum.number_of_frequencies,
            number_of_directions=spectrum.number_of_directions,
            number_of_points=spectrum.number_of_spectra,
        )
        return DataArray(
            data=dissipation,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

    def rate(self, spectrum: FrequencyDirectionSpectrum, memoize=None) -> DataArray:
        if memoize is None:
            memoize = {}

        dissipation = st4_dissipation(
            variance_density=spectrum.variance_density.values,
            depth=spectrum.depth.values,
            radian_frequency=spectrum.radian_frequency.values,
            radian_direction=spectrum.radian_direction.values,
            direction_step=spectrum.direction_step.values,
            frequency_step=spectrum.frequency_step.values,
            saturation_breaking_constant=self.saturation_breaking_constant,
            saturation_breaking_directional_control=self.saturation_breaking_directional_control,
            saturation_cosine_power=self.saturation_cosine_power,
            saturation_integration_width_degrees=self.saturation_integration_width_degrees,
            saturation_threshold=self.saturation_threshold,
            cumulative_breaking_constant=self.cumulative_breaking_constant,
            cumulative_breaking_max_relative_frequency=self.cumulative_breaking_max_relative_frequency,
            number_of_frequencies=spectrum.number_of_frequencies,
            number_of_directions=spectrum.number_of_directions,
            number_of_points=spectrum.number_of_spectra,
        )
        return DataArray(data=dissipation, dims=spectrum.dims, coords=spectrum.coords())


@njit(cache=True)
def st4_dissipation(
    variance_density,
    depth,
    radian_frequency,
    radian_direction,
    direction_step,
    frequency_step,
    saturation_breaking_constant,
    saturation_breaking_directional_control,
    saturation_cosine_power,
    saturation_integration_width_degrees,
    saturation_threshold,
    cumulative_breaking_constant,
    cumulative_breaking_max_relative_frequency,
    number_of_frequencies,
    number_of_directions,
    number_of_points,
):
    dissipation = empty(
        (number_of_points, number_of_frequencies, number_of_directions), "float64"
    )
    for point_index in range(number_of_points):
        dissipation_breaking = st4_dissipation_breaking(
            variance_density=variance_density[point_index, :, :],
            depth=depth[point_index],
            radian_frequency=radian_frequency,
            radian_direction=radian_direction,
            direction_step=direction_step,
            frequency_step=frequency_step,
            saturation_breaking_constant=saturation_breaking_constant,
            saturation_breaking_directional_control=saturation_breaking_directional_control,
            saturation_cosine_power=saturation_cosine_power,
            saturation_integration_width_degrees=saturation_integration_width_degrees,
            saturation_threshold=saturation_threshold,
            cumulative_breaking_constant=cumulative_breaking_constant,
            cumulative_breaking_max_relative_frequency=cumulative_breaking_max_relative_frequency,
            number_of_frequencies=number_of_frequencies,
            number_of_directions=number_of_directions,
        )
        dissipation[point_index, :, :] = dissipation_breaking
    return dissipation


@njit(cache=True)
def st4_bulk_dissipation(
    variance_density,
    depth,
    radian_frequency,
    radian_direction,
    direction_step,
    frequency_step,
    saturation_breaking_constant,
    saturation_breaking_directional_control,
    saturation_cosine_power,
    saturation_integration_width_degrees,
    saturation_threshold,
    cumulative_breaking_constant,
    cumulative_breaking_max_relative_frequency,
    number_of_frequencies,
    number_of_directions,
    number_of_points,
):
    dissipation = empty(number_of_points, "float64")
    for point_index in range(number_of_points):
        dissipation_breaking = st4_dissipation_breaking(
            variance_density=variance_density[point_index, :, :],
            depth=depth[point_index],
            radian_frequency=radian_frequency,
            radian_direction=radian_direction,
            direction_step=direction_step,
            frequency_step=frequency_step,
            saturation_breaking_constant=saturation_breaking_constant,
            saturation_breaking_directional_control=saturation_breaking_directional_control,
            saturation_cosine_power=saturation_cosine_power,
            saturation_integration_width_degrees=saturation_integration_width_degrees,
            saturation_threshold=saturation_threshold,
            cumulative_breaking_constant=cumulative_breaking_constant,
            cumulative_breaking_max_relative_frequency=cumulative_breaking_max_relative_frequency,
            number_of_frequencies=number_of_frequencies,
            number_of_directions=number_of_directions,
        )
        grid_info = {"frequency_step": frequency_step, "direction_step": direction_step}
        dissipation[point_index] = numba_integrate_spectral_data(
            dissipation_breaking, grid_info
        )
    return dissipation


@njit(cache=True)
def st4_band_integrated_saturation(
    variance_density: NDArray,
    group_velocity: NDArray,
    wavenumber: NDArray,
    radian_direction: NDArray,
    direction_step: NDArray,
    number_of_frequencies: int,
    number_of_directions: int,
    integration_width_degrees: int,
    cosine_power=2,
):
    directional_saturation_spec = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )
    band_integrated_saturation = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )
    integration_width_radians = integration_width_degrees * pi / 180

    for frequency_index in range(number_of_frequencies):
        directional_saturation_spec[frequency_index, :] = (
            variance_density[frequency_index, :]
            * group_velocity[frequency_index]
            * wavenumber[frequency_index] ** 3
            / 2
            / pi
        )

    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            integrant = 0.0
            for dummy_direction_index in range(number_of_directions):
                mutual_angle = (
                    radian_direction[dummy_direction_index]
                    - radian_direction[direction_index]
                    + pi
                ) % (2 * pi) - pi
                if abs(mutual_angle) > integration_width_radians:
                    continue

                integrant += (
                    directional_saturation_spec[frequency_index, dummy_direction_index]
                    * cos(mutual_angle) ** cosine_power
                    * direction_step[dummy_direction_index]
                )
            band_integrated_saturation[frequency_index, direction_index] = integrant
    return band_integrated_saturation


@njit(cache=True)
def st4_cumulative_breaking(
    variance_density: NDArray,
    saturation: NDArray,
    radian_frequency: NDArray,
    group_velocity: NDArray,
    wave_speed: NDArray,
    radian_direction: NDArray,
    direction_step: NDArray,
    frequency_step: NDArray,
    saturation_threshold: float,
    cumulative_breaking_constant: float,
    cumulative_breaking_max_relative_frequency: float,
    number_of_frequencies: int,
    number_of_directions: int,
):
    """

    :param saturation:
    :param radian_frequency:
    :param group_velocity:
    :param wave_speed:
    :param radian_direction:
    :param direction_step:
    :param frequency_step:
    :param saturation_threshold:
    :param cumulative_breaking_max_relative_frequency:
    :param number_of_frequencies:
    :param number_of_directions:
    :return:
    """

    jacobian_radian_to_degrees_direction = pi / 180
    jacobian_radian_to_hertz_frequency = 2 * pi
    jacobian_wavenumber_to_radian_frequency = 1 / group_velocity

    jacobian = (
        jacobian_radian_to_hertz_frequency
        * jacobian_wavenumber_to_radian_frequency
        * jacobian_radian_to_degrees_direction
    )

    cosines = cos(radian_direction)
    sines = sin(radian_direction)

    threshold = sqrt(saturation) - sqrt(saturation_threshold)

    cumulative_breaking = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )
    wave_speed_east = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )
    wave_speed_north = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )

    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            wave_speed_east[frequency_index, direction_index] = (
                cosines[direction_index] * wave_speed[frequency_index]
            )
            wave_speed_north[frequency_index, direction_index] = (
                sines[direction_index] * wave_speed[frequency_index]
            )

    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            # Initialize
            strength = 0.0

            for dummy_frequency_index in range(0, number_of_frequencies):
                #
                if (
                    radian_frequency[dummy_frequency_index]
                    > radian_frequency[frequency_index]
                    * cumulative_breaking_max_relative_frequency
                ):
                    break

                for dummy_direction_index in range(0, number_of_directions):
                    if threshold[dummy_frequency_index, dummy_direction_index] <= 0:
                        continue

                    integrant = (
                        threshold[dummy_frequency_index, dummy_direction_index] ** 2
                        * jacobian[dummy_frequency_index]
                    )

                    # Calculate the celerity magnitude
                    delta_wave_speed_east = (
                        wave_speed_east[dummy_frequency_index, dummy_direction_index]
                        - wave_speed_east[frequency_index, direction_index]
                    )
                    delta_wave_speed_north = (
                        wave_speed_north[dummy_frequency_index, dummy_direction_index]
                        - wave_speed_north[frequency_index, direction_index]
                    )
                    delta_speed_magnitude = sqrt(
                        delta_wave_speed_east**2 + delta_wave_speed_north**2
                    )

                    strength += (
                        frequency_step[dummy_frequency_index]
                        * direction_step[dummy_direction_index]
                        * delta_speed_magnitude
                        * integrant
                    )

            cumulative_breaking[frequency_index, direction_index] = (
                -1.44
                * cumulative_breaking_constant
                * strength
                * variance_density[frequency_index, direction_index]
            )
    return cumulative_breaking


@njit(cache=True)
def st4_dissipation_breaking(
    variance_density: NDArray,
    depth: NDArray,
    radian_frequency: NDArray,
    radian_direction: NDArray,
    direction_step: NDArray,
    frequency_step: NDArray,
    saturation_breaking_constant,
    saturation_breaking_directional_control,
    saturation_cosine_power,
    saturation_integration_width_degrees,
    saturation_threshold: float,
    cumulative_breaking_constant: float,
    cumulative_breaking_max_relative_frequency: float,
    number_of_frequencies: int,
    number_of_directions: int,
):
    wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    wave_speed = radian_frequency / wavenumber
    group_velocity = intrinsic_group_velocity(wavenumber, depth)

    band_integrated_saturation = st4_band_integrated_saturation(
        variance_density=variance_density,
        group_velocity=group_velocity,
        wavenumber=wavenumber,
        radian_direction=radian_direction,
        direction_step=direction_step,
        number_of_frequencies=number_of_frequencies,
        number_of_directions=number_of_directions,
        integration_width_degrees=saturation_integration_width_degrees,
        cosine_power=saturation_cosine_power,
    )

    cumulative_breaking = st4_cumulative_breaking(
        variance_density=variance_density,
        saturation=band_integrated_saturation,
        radian_frequency=radian_frequency,
        group_velocity=group_velocity,
        wave_speed=wave_speed,
        radian_direction=radian_direction,
        direction_step=direction_step,
        frequency_step=frequency_step,
        saturation_threshold=saturation_threshold,
        cumulative_breaking_constant=cumulative_breaking_constant,
        cumulative_breaking_max_relative_frequency=cumulative_breaking_max_relative_frequency,
        number_of_frequencies=number_of_frequencies,
        number_of_directions=number_of_directions,
    )

    saturation_breaking = st4_saturation_breaking(
        variance_density=variance_density,
        band_integrated_saturation=band_integrated_saturation,
        radian_frequency=radian_frequency,
        number_of_frequencies=number_of_frequencies,
        number_of_directions=number_of_directions,
        saturation_breaking_constant=saturation_breaking_constant,
        saturation_breaking_directional_control=saturation_breaking_directional_control,
        saturation_threshold=saturation_threshold,
    )

    return cumulative_breaking + saturation_breaking


@njit(cache=True)
def st4_saturation_breaking(
    variance_density,
    band_integrated_saturation,
    radian_frequency,
    number_of_frequencies,
    number_of_directions,
    saturation_breaking_constant,
    saturation_breaking_directional_control,
    saturation_threshold,
):
    saturation_breaking = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )

    for frequency_index in range(number_of_frequencies):
        maximum_band_integrated_saturation = max(
            band_integrated_saturation[frequency_index, :]
        )
        isotropic_saturation_exceedence = (
            (maximum_band_integrated_saturation - saturation_threshold)
            / saturation_threshold
            * saturation_breaking_directional_control
        )
        if isotropic_saturation_exceedence < 0.0:
            isotropic_saturation_exceedence = 0.0

        for direction_index in range(number_of_directions):
            directional_saturation_exceedence = (
                (
                    band_integrated_saturation[frequency_index, direction_index]
                    - saturation_threshold
                )
                / saturation_threshold
                * (1 - saturation_breaking_directional_control)
            )
            if directional_saturation_exceedence < 0.0:
                directional_saturation_exceedence = 0.0

            relative_exceedence_level = (
                isotropic_saturation_exceedence + directional_saturation_exceedence
            )

            saturation_breaking[frequency_index, direction_index] = (
                -saturation_breaking_constant
                * relative_exceedence_level**2
                * radian_frequency[frequency_index]
                * variance_density[frequency_index, direction_index]
            )
    return saturation_breaking
