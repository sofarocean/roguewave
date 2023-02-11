from typing import TypedDict
from roguewave.wavephysics.balance import Dissipation
from numpy.typing import NDArray
from numpy import pi, abs, cos, sin, sqrt, empty, max, exp
from numba import njit
from roguewave.wavetheory import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_velocity,
)
from roguewave.wavephysics.fluidproperties import (
    GRAVITATIONAL_ACCELERATION
)
from roguewave.wavespectra.operations import numba_directionally_integrate_spectral_data


class RomeroWaveBreakingParameters(TypedDict):
    saturation_breaking_constant: float
    saturation_threshold: float
    saturation_integrated_threshold: float
    breaking_probability_constant: float
    gravitational_acceleration: float


class RomeroWaveBreaking(Dissipation):
    name = "st4 dissipation"

    def __init__(self, parameters: RomeroWaveBreakingParameters = None):
        super(RomeroWaveBreaking, self).__init__(parameters)
        self._dissipation_function = romero_dissipation_breaking

    @staticmethod
    def default_parameters() -> RomeroWaveBreakingParameters:
        return RomeroWaveBreakingParameters(
            saturation_breaking_constant=2.5,
            saturation_threshold=0.005,
            saturation_integrated_threshold=0.0011,
            breaking_probability_constant=3.5e-5,
            gravitational_acceleration = GRAVITATIONAL_ACCELERATION
        )


@njit(cache=True)
def romero_saturation(
    variance_density: NDArray,
    group_velocity: NDArray,
    wavenumber: NDArray,
    number_of_frequencies: int,
    number_of_directions: int,
):
    directional_saturation_spec = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )

    for frequency_index in range(number_of_frequencies):
        directional_saturation_spec[frequency_index, :] = (
            variance_density[frequency_index, :]
            * group_velocity[frequency_index]
            * wavenumber[frequency_index] ** 3
            / 2
            / pi
        )

    return directional_saturation_spec


@njit(cache=True)
def romero_dissipation_breaking(
    variance_density: NDArray, depth: NDArray, spectral_grid, parameters
):
    number_of_directions = variance_density.shape[1]
    number_of_frequencies = variance_density.shape[0]

    radian_frequency = spectral_grid["radian_frequency"]
    saturation_threshold = parameters["saturation_threshold"]
    saturation_breaking_constant = parameters["saturation_breaking_constant"]
    saturation_integrated_threshold = parameters["saturation_integrated_threshold"]
    breaking_probability_constant = parameters['breaking_probability_constant']
    gravitational_acceleration = parameters['gravitational_acceleration']

    wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    wave_speed = radian_frequency / wavenumber
    group_velocity = intrinsic_group_velocity(wavenumber, depth)

    directionaL_saturation = romero_saturation(
        variance_density=variance_density,
        group_velocity=group_velocity,
        wavenumber=wavenumber,
        number_of_frequencies=number_of_frequencies,
        number_of_directions=number_of_directions,
    )
    saturation = numba_directionally_integrate_spectral_data(directionaL_saturation,spectral_grid)
    dissipation_rate_per_unit_breaking_crest_length = empty((number_of_frequencies))

    for frequency_index in range(number_of_frequencies):
        delta = sqrt(saturation[frequency_index]) - sqrt(saturation_integrated_threshold)
        if delta > 0.0:
            dissipation_rate_per_unit_breaking_crest_length[frequency_index] = (
                saturation_breaking_constant
                * delta**2.5
                / gravitational_acceleration**2
            )
        else:
            dissipation_rate_per_unit_breaking_crest_length[frequency_index] = 0.0

    probability = breaking_probability(
        directionaL_saturation,
        wavenumber,
        saturation_threshold,
        breaking_probability_constant,
        number_of_frequencies,
        number_of_directions
    )

    saturation_breaking = empty((number_of_frequencies, number_of_directions))

    jacobian = 2*pi/group_velocity
    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            saturation_breaking[frequency_index,direction_index] = -(
                jacobian[frequency_index]
                * dissipation_rate_per_unit_breaking_crest_length[frequency_index]
                * probability[frequency_index,direction_index]
                * wave_speed[frequency_index]**5
                / gravitational_acceleration**2
            )

    return saturation_breaking

@njit(cache=True)
def breaking_probability(
    directional_saturation,
    wavenumber,
    saturation_threshold,
    breaking_probability_constant,
    number_of_frequencies,
    number_of_directions,
):
    probability = empty(
        (number_of_frequencies, number_of_directions), dtype="float64"
    )

    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            if wavenumber[frequency_index] > 0.0:
                probability[frequency_index,direction_index] = (
                    breaking_probability_constant
                    * exp(
                        -saturation_threshold
                        / directional_saturation[frequency_index,direction_index]
                    )
                    / wavenumber[frequency_index]
                )
            else:
                probability[frequency_index] = 0.0

    return probability