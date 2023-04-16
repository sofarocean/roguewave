import numpy
from numba import njit
from scipy.interpolate import make_interp_spline, CubicSpline
from roguewave.interpolate.spline import cubic_spline

from roguewave.tools.grid import midpoint_rule_step
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
        ) + 1.0 / 4.0 * transition_frequencies[index] * (
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


def _cdf_interpolate(
    interpolation_frequency: numpy.ndarray,
    frequency: numpy.ndarray,
    frequency_spectrum: numpy.ndarray,
    interpolating_spline_order: int = 3,
    positive: bool = False,
    frequency_axis=-1,
    **kwargs
) -> numpy.ndarray:
    """
    Interpolate the spectrum using the cdf.

    :param interpolation_frequency: frequencies to estimate the spectrum at.
    :param frequency: Frequencies of the spectrum. Shape = ( nf, )
    :param frequency_spectrum: Frequency Variance density spectrum. Shape = ( np , nf )
    :param interpolating_spline_order: Order of the spline to use in the interpolation (max 5 supported by scipy)
    :param positive: Ensure the output is positive (e.g. for A1 or B1 densities output need not be strictly positive).
    :return:
    """
    #
    frequency_step = midpoint_rule_step(frequency)
    integration_frequencies = numpy.concatenate(([0], numpy.cumsum(frequency_step)))
    integration_frequencies = (
        integration_frequencies - frequency_step[0] / 2 + frequency[0]
    )

    cumsum = numpy.cumsum(frequency_spectrum * frequency_step, axis=frequency_axis)
    shape = list(cumsum.shape)
    shape[frequency_axis] = 1
    cumsum = numpy.concatenate((numpy.zeros(shape), cumsum), axis=-1)

    if interpolating_spline_order == 3:
        #
        interpolator = cubic_spline(
            integration_frequencies, cumsum, monotone_interpolation=positive
        )
    else:
        interpolator = make_interp_spline(
            integration_frequencies, cumsum, k=interpolating_spline_order, axis=-1
        )

    return interpolator.derivative()(interpolation_frequency)


def spline_peak_frequency(
    frequency: numpy.ndarray, frequency_spectrum: numpy.ndarray, frequency_axis=-1
) -> numpy.ndarray:
    """
    Estimate the peak frequency of the spectrum based on a cubic spline interpolation of the partially integrated
    variance.

    :param frequency: Frequencies of the spectrum. Shape = ( nf, )
    :param frequency_spectrum: Frequency Variance density spectrum. Shape = ( np , nf )
    :return: peak frequencies. Shape = ( np, )
    """
    #

    # Calculate the binsize for each of the frequency bins. We assume that the given frequencies represent the center
    # of a bin, and that the bin width at frequency i is determined as the sum of half the up and downwind differences:
    #
    #  frequency_step[i]   =  (frequency_step[i] - frequency_step[i-1])/2 + (frequency_step[i+1] - frequency_step[i])/2
    #
    # At boundaries we simply take twice the up or downwind difference, e.g.:
    #
    # frequency_step[0] = (frequency_step[1] - frequency_step[0])
    #
    frequency_step = midpoint_rule_step(frequency)

    # Whereas the given frequencies represent the center, we assume that cumulative function is sampled at the bin
    # edges. First create the cumulative sum frequency relative to the start...
    integration_frequencies = numpy.concatenate(([0], numpy.cumsum(frequency_step)))

    # and then add the origin (first bin centered frequency minus half the step size.
    integration_frequencies = (
        integration_frequencies - frequency_step[0] / 2 + frequency[0]
    )

    # Since spectra are typicall given as densities, muliply with the step to get the bin integrated values.
    bin_integrated_values = frequency_spectrum * frequency_step

    # calculate the cumulative function
    cumsum = numpy.cumsum(bin_integrated_values, axis=frequency_axis)

    shape = list(cumsum.shape)
    shape[frequency_axis] = 1

    # Add leading 0s as at the first frequency the integration is 0.
    cumsum = numpy.concatenate((numpy.zeros(shape), cumsum), axis=frequency_axis)

    # Construct a cubic spline interpolator, and then differentiate to get the density function.
    interpolator = cubic_spline(
        integration_frequencies, cumsum, monotone_interpolation=True
    ).derivative()  # CubicSpline(integration_frequencies, cumsum, axis=frequency_axis).derivative()

    # Find the maxima of the density function by setting dEdf = 0, and finding the roots of all the splines representing
    # the density function.
    list_of_roots_for_all_spectra = interpolator.derivative().roots()

    if len(shape) == 1:
        list_of_roots_for_all_spectra = [list_of_roots_for_all_spectra]

    # initialize output memory
    peak_frequency = numpy.zeros(len(list_of_roots_for_all_spectra))

    # We now have a list in which each entry contains all the roots for that given spectrum. Loop over each spectrum
    # and..
    for index, root in enumerate(list_of_roots_for_all_spectra):
        # ..evaluate density spectrum at those roots.
        values_at_roots = interpolator(root)

        # Because the interpolator returns values at the current roots evaluated at _all_ spectra, we still have to
        # select the values at the spectrum of interest. This implementation is silly, adds computational costs, and can
        # probably be improved. It seems "fast enough" so that I'll punt that to another time.
        if len(list_of_roots_for_all_spectra) > 1:
            values_at_roots = values_at_roots[index, :]

        _root = root[numpy.isfinite(values_at_roots)]
        values_at_roots = values_at_roots[numpy.isfinite(values_at_roots)]

        # ... get the root that corresponds to the largest peak.
        index_peak = numpy.argmax(values_at_roots)
        peak_frequency[index] = _root[index_peak]

    return peak_frequency


def make_monotone(spline: CubicSpline) -> CubicSpline:
    """
    Checks a 3rd order spline for monoticity; if the spline not monotone on an interval nodes[ii], nodes[ii+1] we
    substitute the spline on that interval with a simple linear function. Crude, but effective way to correct a
    natural spline.

    Note that the underlying assumption is that the anchor points of the spline are monotone.

    The monoticiy check is based on FC1980:
       Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation.
       SIAM Journal on Numerical Analysis, 17(2), 238-246.

    :param spline: Instance of scipy.interpolate.CubicSpline interpolator class

    :return: a new instance of a CubicSpine that is almost equivalent to the input spline, but corrected to be monotone.
    """

    # Get the nodes and coefficients of the spline. We will be modifying these, so a copy is needed to preserve the
    # original spline.
    spline_nodes = spline.x.copy()
    spline_coefficients = spline.c.copy()

    # Get the interval length for the different splines.
    step = numpy.diff(spline_nodes)

    # Calculate the cumulitive distribution function at the start (left) and end (right) of the spline interval.
    # ---
    right_cdf = numpy.zeros(spline_coefficients.shape[1:])
    left_cdf = spline_coefficients[3, ...]
    right_cdf[:-1] = spline_coefficients[3, 1:, ...]
    for kk in range(0, 4):
        right_cdf[-1] += (step[-1]) ** (3 - kk) * spline_coefficients[kk, -1, ...]

    # Calculate the pdf (derivative) at the start (left) and end (right) of the spline interval.
    # ---
    left_pdf = spline_coefficients[2, ...]
    right_pdf = numpy.zeros(spline_coefficients.shape[1:])

    # indexer to allow for arbitrary trailing dimensions.
    indexer = (..., *([None] * (len(spline_coefficients.shape) - 2)))
    for kk in range(0, 3):
        right_pdf += (3 - kk) * step[indexer] ** (2 - kk) * spline_coefficients[kk, ...]

    #
    # FC1980 shows that if the derivatives at the end point (left_pdf, right_pdf here) are normalized with the
    # secant-line on the interval, giving alpha, beta, a sufficient condition that ensure that the spline is monotone
    # is that aplha*beta > 0 and  alpha**2 + beta**2 <= 9. Note that some splines that violate this condition are
    # monotone- but in general this restricts the monotone region well.
    #

    # For regions where the secant is zero, we set alpha/beta to inf. This avoids issues with dividing by zero.
    secant = (right_cdf - left_cdf) / step[indexer]
    alpha = numpy.full_like(secant, numpy.inf)
    beta = numpy.full_like(secant, numpy.inf)

    mask = secant > 1.0e-10
    alpha[mask] = left_pdf[mask] / secant[mask]
    beta[mask] = right_pdf[mask] / secant[mask]

    critical_circle = alpha**2 + beta**2
    not_monotone = (critical_circle > 9) | (left_pdf * right_pdf < 0)

    # For those splines that are not monotone, set coefs associated with x**3 and x**2 to 0, make the slope equivalent
    # to the secant line. We do not need to update the constant value (4th entry).
    spline_coefficients[0, not_monotone] = 0.0
    spline_coefficients[1, not_monotone] = 0.0
    spline_coefficients[2, not_monotone] = secant[not_monotone]

    # Create a new spline with the given coeficients and return
    return CubicSpline.construct_fast(spline_coefficients, spline_nodes, axis=1)


def make_monotone2(spline: CubicSpline) -> CubicSpline:
    """
    Checks a 3rd order spline for monoticity; if the spline not monotone on an interval nodes[ii], nodes[ii+1] we
    substitute the spline on that interval with a simple linear function. Crude, but effective way to correct a
    natural spline.

    Note that the underlying assumption is that the anchor points of the spline are monotone.

    The monoticiy check is based on FC1980:
       Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation.
       SIAM Journal on Numerical Analysis, 17(2), 238-246.

    :param spline: Instance of scipy.interpolate.CubicSpline interpolator class

    :return: a new instance of a CubicSpine that is almost equivalent to the input spline, but corrected to be monotone.
    """

    # Get the nodes and coefficients of the spline. We will be modifying these, so a copy is needed to preserve the
    # original spline.
    spline_nodes = spline.x.copy()
    spline_coefficients = spline.c.copy()

    # Get the interval length for the different splines.
    step = numpy.diff(spline_nodes)

    # Calculate the cumulitive distribution function at the start (left) and end (right) of the spline interval.
    # ---
    right_cdf = numpy.zeros(spline_coefficients.shape[1:])
    left_cdf = spline_coefficients[3, ...]
    right_cdf[:-1] = spline_coefficients[3, 1:, ...]
    for kk in range(0, 4):
        right_cdf[-1] += (step[-1]) ** (3 - kk) * spline_coefficients[kk, -1, ...]

    # Calculate the pdf (derivative) at the start (left) and end (right) of the spline interval.
    # ---
    left_pdf = spline_coefficients[2, ...]
    right_pdf = numpy.zeros(spline_coefficients.shape[1:])

    # indexer to allow for arbitrary trailing dimensions.
    indexer = (..., *([None] * (len(spline_coefficients.shape) - 2)))
    for kk in range(0, 3):
        right_pdf += (3 - kk) * step[indexer] ** (2 - kk) * spline_coefficients[kk, ...]

    #
    # FC1980 shows that if the derivatives at the end point (left_pdf, right_pdf here) are normalized with the
    # secant-line on the interval, giving alpha, beta, a sufficient condition that ensure that the spline is monotone
    # is that aplha*beta > 0 and  alpha**2 + beta**2 <= 9. Note that some splines that violate this condition are
    # monotone- but in general this restricts the monotone region well.
    #

    # For regions where the secant is zero, we set alpha/beta to inf. This avoids issues with dividing by zero.
    secant = (right_cdf - left_cdf) / step[indexer]
    alpha = numpy.full_like(secant, numpy.inf)
    beta = numpy.full_like(secant, numpy.inf)

    mask = secant > 1.0e-10
    alpha[mask] = left_pdf[mask] / secant[mask]
    beta[mask] = right_pdf[mask] / secant[mask]

    critical_circle = alpha**2 + beta**2
    not_monotone = (critical_circle > 9) | (left_pdf * right_pdf < 0)

    # For those splines that are not monotone, set coefs associated with x**3 and x**2 to 0, make the slope equivalent
    # to the secant line. We do not need to update the constant value (4th entry).

    nf = numpy.shape(left_cdf)[0]
    for jf in range(0, nf):

        delta = step[jf]

        mask = not_monotone[jf, ...]
        if numpy.any(mask):
            if jf == 0 or jf == nf - 1:
                fd1 = secant[jf, mask]
                fd2 = secant[jf, mask]
            else:
                fd1 = (
                    3
                    * secant[jf - 1, mask]
                    * secant[jf, mask]
                    / (2 * secant[jf - 1, mask] + secant[jf, mask])
                )
                fd2 = (
                    3
                    * secant[jf + 1, mask]
                    * secant[jf, mask]
                    / (secant[jf + 1, mask] + 2 * secant[jf, mask])
                )

            spline_coefficients = update_spline(
                spline_coefficients, jf, fd1, fd2, delta, mask, recursive=False
            )

    # Create a new spline with the given coeficients and return
    return CubicSpline.construct_fast(spline_coefficients, spline_nodes, axis=1)


def update_spline(spline_coefficients, jf, fd1, fd2, delta, mask, recursive=False):

    f1 = spline_coefficients[3, jf, mask]
    f2 = (
        spline_coefficients[0, jf, mask] * delta**3
        + spline_coefficients[1, jf, mask] * delta**2
        + spline_coefficients[2, jf, mask] * delta
        + spline_coefficients[3, jf, mask]
    )
    if fd1 is None:
        fd1 = spline_coefficients[2, jf, mask]
    else:
        if recursive:
            if jf > 1:
                spline_coefficients = update_spline(
                    spline_coefficients, jf - 1, None, fd1, delta, mask, recursive=False
                )

    if fd2 is None:
        fd2 = (
            3 * spline_coefficients[0, jf, mask] * delta**2
            + 2 * spline_coefficients[1, jf, mask] * delta
            + spline_coefficients[2, jf, mask]
        )
    else:
        if recursive:
            if jf < spline_coefficients.shape[1] - 1:
                spline_coefficients = update_spline(
                    spline_coefficients, jf + 1, fd2, None, delta, mask, recursive=False
                )

    d = f1
    c = fd1
    a = fd2 / delta**2 - 2 * f2 / delta**3 + c / delta**2 + 2 * d / delta**3
    b = 3 * f2 / delta**2 - fd2 / delta - 3 * d / delta**2 - 2 * c / delta
    spline_coefficients[0, jf, mask] = a
    spline_coefficients[1, jf, mask] = b
    spline_coefficients[2, jf, mask] = c

    return spline_coefficients
