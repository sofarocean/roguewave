import numpy
from numba import njit
from roguewave.tools.time_integration import integrated_response_factor_spectral_tail


@njit(cache=True)
def _fit(x, y, power):
    X = x**power
    coef = numpy.sum(X * y) / numpy.sum(X * X)
    goodness_of_fit = numpy.sum((coef * X - y) ** 2)
    return coef, goodness_of_fit


@njit(cache=True)
def tail_fit(x, y, power):
    if power is None:
        coef_4, goodness_of_fit_4 = _fit(x, y, -4)
        coef_5, goodness_of_fit_5 = _fit(x, y, -5)
        if goodness_of_fit_5 < goodness_of_fit_4:
            return coef_5, -5
        else:
            return coef_4, -4

    else:
        coef, goodness_of_fit = _fit(x, y, power)
        return coef, power


@njit(cache=True)
def transition_frequency(
    energy_last_resoved_frequency, last_resolved_frequency, integrated_tail
):
    """
    Estimate the transition frequency from a -4 to -5 tail given the last resolved frequency and the integrated energy
    in the tail.

    :param energy_last_resoved_frequency:
    :param last_resolved_frequency:
    :param integrated_tail:
    :return:
    """
    return (
        (energy_last_resoved_frequency * last_resolved_frequency**4)
        / (
            4 * energy_last_resoved_frequency * energy_last_resoved_frequency
            - 12 * integrated_tail
        )
    ) ** (1 / 3)


@njit(cache=True)
def numba_fill_zeros_or_nan_in_tail(
    variance_density: numpy.ndarray,
    frequencies: numpy.ndarray,
    power=None,
    zero=0.0,
    points_in_fit=10,
    tail_information=None,
):
    input_shape = variance_density.shape
    number_of_elements = 1
    for value in input_shape[:-1]:
        number_of_elements *= value

    number_of_frequencies = input_shape[-1]
    variance_density = variance_density.reshape((number_of_elements, input_shape[-1]))

    if tail_information is None:
        tail_energy = numpy.zeros(number_of_elements)
        tail_bounds = (frequencies[-1] + 1, frequencies[-1] + 2)
    else:
        tail_energy = tail_information[1]
        tail_bounds = tail_information[0]

    for ii in range(0, number_of_elements):

        for jj in range(number_of_frequencies - 1, -1, -1):
            if variance_density[ii, jj] > zero:
                # Note, for nan values this is also always false. No need to seperately check for that.
                index = jj
                break
        else:
            # no valid value found, we cannot extrapolate. Technically this is not needed as we catch this below as
            # well (since index=number_of_frequencies by default). But to make more explicit that we catch for this
            # scenario I leave it in.
            continue

        if index == number_of_frequencies - 1:
            continue
        elif index < points_in_fit:
            continue

        last_resolved_frequency = frequencies[index]
        last_resolved_energy = variance_density[ii, index]

        if tail_energy[ii] > 0.0:
            trans_freq, starting_energy = _compound_tail(
                last_resolved_frequency,
                last_resolved_energy,
                tail_energy[ii],
                tail_bounds,
            )
            for jj in range(index + 1, number_of_frequencies):
                if frequencies[jj] >= trans_freq:
                    variance_density[ii, jj] = (
                        starting_energy * trans_freq * frequencies[jj] ** -5
                    )
                else:
                    variance_density[ii, jj] = starting_energy * (frequencies[jj]) ** -4
        else:
            coef, fitted_power = tail_fit(
                x=frequencies[index - points_in_fit + 1 : index + 1],
                y=variance_density[ii, index - points_in_fit + 1 : index + 1],
                power=power,
            )

            for jj in range(index + 1, number_of_frequencies):
                variance_density[ii, jj] = coef * (frequencies[jj]) ** fitted_power

    return numpy.reshape(variance_density, input_shape)


@njit(cache=True)
def _compound_tail(
    last_resolved_frequency, last_resolved_energy, raw_tail_energy, tail_bounds
):
    transition_frequencies = numpy.linspace(tail_bounds[0], tail_bounds[1], 11)

    goodness_of_fit = 0.0
    transition_frequency = transition_frequencies[0]
    starting_energy = 0.0
    for index in range(transition_frequencies.shape[0]):
        tail_energy = raw_tail_energy * integrated_response_factor_spectral_tail(
            -4,
            tail_bounds[0],
            tail_bounds[1],
            2.5,
            transition_frequency=transition_frequencies[index],
        )

        freq_int = 1.0 / 3.0 * (
            tail_bounds[0] ** -3 - transition_frequencies[index] ** -3
        ) + 1 / 4 * transition_frequencies[index] * (
            transition_frequencies[index] ** -4 - tail_bounds[1] ** -4
        )
        current_starting_energy = tail_energy / freq_int
        current_fit = (
            current_starting_energy * last_resolved_frequency**-4
            - last_resolved_energy
        ) ** 2

        if index == 0:
            goodness_of_fit = (
                current_starting_energy * last_resolved_frequency**-5
                - last_resolved_energy
            ) ** 2
            starting_energy = current_starting_energy
            transition_frequency = transition_frequencies[index]

        else:
            if current_fit < goodness_of_fit:
                goodness_of_fit = current_fit
                starting_energy = current_starting_energy
                transition_frequency = transition_frequencies[index]

    return transition_frequency, starting_energy


@njit(cache=True)
def _starting_energy(raw_tail_energy, fitted_power, last_resolved_frequency):
    tail_energy = raw_tail_energy * integrated_response_factor_spectral_tail(
        fitted_power, last_resolved_frequency, 0.8, 2.5
    )
    starting_energy = tail_energy / (
        (1 / (fitted_power + 1))
        * (0.8 ** (fitted_power + 1) - last_resolved_frequency ** (fitted_power + 1))
    )
    return starting_energy


@njit(cache=True)
def find_starting_energy(raw_tail_energy, fitted_power, last_resolved_frequency):
    tail_energy = raw_tail_energy * integrated_response_factor_spectral_tail(
        fitted_power, last_resolved_frequency, 0.8, 2.5
    )
    starting_energy = tail_energy / (
        (1 / (fitted_power + 1))
        * (0.8 ** (fitted_power + 1) - last_resolved_frequency ** (fitted_power + 1))
    )
    return starting_energy
