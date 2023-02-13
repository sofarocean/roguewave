import numpy
from numba import njit


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
def fill_zeros_or_nan_in_tail(
    variance_density: numpy.ndarray,
    frequencies: numpy.ndarray,
    power=None,
    zero=0.0,
    points_in_fit=10,
):

    input_shape = variance_density.shape
    number_of_elements = 1
    for value in input_shape[:-1]:
        number_of_elements *= value

    number_of_frequencies = input_shape[-1]
    variance_density = variance_density.reshape((number_of_elements, input_shape[-1]))

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

        coef, fitted_power = tail_fit(
            x=frequencies[index - points_in_fit + 1 : index + 1],
            y=variance_density[ii, index - points_in_fit + 1 : index + 1],
            power=power,
        )

        for jj in range(index + 1, number_of_frequencies):
            variance_density[ii, jj] = coef * (frequencies[jj]) ** fitted_power

    return numpy.reshape(variance_density, input_shape)
