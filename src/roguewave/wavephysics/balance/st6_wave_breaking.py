from numpy import pi, empty
from typing import TypedDict
from roguewave.wavephysics.balance import Dissipation
from roguewave.wavespectra.operations import numba_directionally_integrate_spectral_data
from roguewave.wavetheory.lineardispersion import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_velocity,
)
from numba import njit


class ST6WaveBreakingParameters(TypedDict):
    p1: float
    p2: float
    a1: float
    a2: float
    saturation_threshold: float


class ST6WaveBreaking(Dissipation):
    name = "st6 dissipation"

    def __init__(self, parameters: ST6WaveBreakingParameters = None):
        super(ST6WaveBreaking, self).__init__(parameters)
        self._dissipation_function = st6_dissipation

    @staticmethod
    def default_parameters() -> ST6WaveBreakingParameters:
        return ST6WaveBreakingParameters(
            p1=4, p2=4, a1=4.75 * 10**-6, a2=7e-5, saturation_threshold=0.035**2
        )


@njit(cache=True)
def st6_dissipation(variance_density, depth, spectral_grid, parameters):
    number_of_frequencies = variance_density.shape[0]

    radian_frequency = spectral_grid["radian_frequency"]
    saturation_threshold = parameters["saturation_threshold"]

    frequency_spectrum = numba_directionally_integrate_spectral_data(
        variance_density, spectral_grid
    )
    wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    group_velocity = intrinsic_group_velocity(wavenumber, depth)
    saturation_spectrum = frequency_spectrum * group_velocity * wavenumber**3 / 2 / pi

    relative_saturation_exceedence = empty(number_of_frequencies)
    for frequency_index in range(number_of_frequencies):
        relative_exceedence = (
            saturation_spectrum[frequency_index] - saturation_threshold
        ) / saturation_threshold
        if relative_exceedence > 0.0:
            relative_saturation_exceedence[frequency_index] = relative_exceedence
        else:
            relative_saturation_exceedence[frequency_index] = 0.0

    inherent = st6_inherent(
        variance_density, relative_saturation_exceedence, spectral_grid, parameters
    )
    cumulative = st6_cumulative(
        variance_density, relative_saturation_exceedence, spectral_grid, parameters
    )

    return inherent + cumulative


@njit(cache=True)
def st6_inherent(
    variance_density, relative_saturation_exceedence, spectral_grid, parameters
):
    number_of_frequencies = variance_density.shape[0]
    number_of_directions = variance_density.shape[1]

    frequency = spectral_grid["radian_frequency"] / 2.0 / pi
    a1 = parameters["a1"]
    p1 = parameters["p1"]

    inherent = empty((number_of_frequencies, number_of_directions))
    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            inherent[frequency_index, direction_index] = (
                -a1
                * relative_saturation_exceedence[frequency_index] ** p1
                * frequency[frequency_index]
                * variance_density[frequency_index, direction_index]
            )
    return inherent


@njit(cache=True)
def st6_cumulative(
    variance_density, relative_saturation_exceedence, spectral_grid, parameters
):
    number_of_frequencies = variance_density.shape[0]
    number_of_directions = variance_density.shape[1]
    frequency_step = spectral_grid["frequency_step"]
    a2 = parameters["a2"]
    p2 = parameters["p2"]

    cumulative = empty((number_of_frequencies, number_of_directions))
    run_sum = 0.0
    for frequency_index in range(number_of_frequencies):
        run_sum += (
            relative_saturation_exceedence[frequency_index]
            * frequency_step[frequency_index]
        )

        for direction_index in range(number_of_directions):
            cumulative[frequency_index, direction_index] = (
                -a2 * run_sum**p2 * variance_density[frequency_index, direction_index]
            )
    return cumulative
