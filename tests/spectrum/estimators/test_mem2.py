from roguewave.wavespectra.parametric import create_parametric_spectrum
import numpy
from roguewave import concatenate_spectra, FrequencyDirectionSpectrum
from datetime import datetime, timezone
from numpy.testing import assert_allclose
from roguewave.wavespectra.estimators.mem2 import (
    mem2_newton_solver,
    initial_value,
    moment_constraints,
    mem2_directional_distribution,
    mem2_jacobian,
)
from roguewave.wavespectra.estimators.utils import get_direction_increment
from scipy.optimize import root


def get_2d_spec() -> FrequencyDirectionSpectrum:
    freq = numpy.linspace(0, 1, 20)
    dir = numpy.linspace(0, 360, 36, endpoint=False)
    time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    spec = create_parametric_spectrum(
        freq, "pm", 0.1, 1, dir, "raised_cosine", 35, 20, depth=numpy.inf, time=time
    )
    return concatenate_spectra([spec], "time")


def get_case(case):
    """
    Some test cases taken from a large data set that proved to have particularly difficult convergence behaviour.
    :param case:
    :return:
    """

    if case == 0:
        moments = [0.557185, -0.795699, -0.305963, -0.884653]
    elif case == 1:
        moments = [-0.564027, -0.505376, -0.231672, 0.471163]
    elif case == 2:
        moments = [-0.533724, 0.751711, -0.27957, -0.808407]
    elif case == 3:
        moments = [0.458456, -0.848485, -0.515151, -0.753666]
    elif case == 4:
        moments = [0.458456 + 0.06, -0.848485, -0.515151, -0.753666]

    directions_radians = numpy.linspace(0, 360, 36, endpoint=False) * numpy.pi / 180
    direction_increment = get_direction_increment(directions_radians)
    twiddle_factors = numpy.zeros((4, 36))
    twiddle_factors[0, :] = numpy.cos(directions_radians)
    twiddle_factors[1, :] = numpy.sin(directions_radians)
    twiddle_factors[2, :] = numpy.cos(2 * directions_radians)
    twiddle_factors[3, :] = numpy.sin(2 * directions_radians)
    return (
        numpy.array(moments),
        directions_radians,
        direction_increment,
        twiddle_factors,
    )


def desired_distribution(icase):
    moments, directions_radians, direction_increment, twiddle_factors = get_case(icase)
    a1, b1, a2, b2 = moments
    guess = initial_value(
        numpy.array(a1), numpy.array(b1), numpy.array(a2), numpy.array(b2)
    )
    res = root(
        moment_constraints,
        guess,
        args=(twiddle_factors, moments, direction_increment),
        method="lm",
    )
    lambas = res.x
    return mem2_directional_distribution(lambas, direction_increment, twiddle_factors)


def test_mem2():
    moments = ["a1", "b1", "a2", "b2"]

    spec2d = get_2d_spec()
    spec1d = spec2d.as_frequency_spectrum()
    reconstructed_scipy = spec1d.as_frequency_direction_spectrum(
        36, method="mem2", solution_method="scipy"
    )
    reconstructed_newton = spec1d.as_frequency_direction_spectrum(
        36, method="mem2", solution_method="newton"
    )
    for moment in moments:
        x = getattr(spec2d, moment)
        y = getattr(reconstructed_scipy, moment)
        assert_allclose(y, x, rtol=1e-2, atol=1e-2)

    for moment in moments:
        x = getattr(spec2d, moment)
        y = getattr(reconstructed_newton, moment)
        assert_allclose(y, x, rtol=1e-2, atol=1e-2)

    for moment in moments:
        x = getattr(reconstructed_scipy, moment)
        y = getattr(reconstructed_newton, moment)
        assert_allclose(y, x, rtol=1e-2, atol=1e-2)
    pass


def test_estimate_distribution_newton():
    for icase in range(0, 4):
        moments, directions_radians, direction_increment, twiddle_factors = get_case(
            icase
        )
        a1, b1, a2, b2 = moments
        guess = initial_value(
            numpy.array(a1), numpy.array(b1), numpy.array(a2), numpy.array(b2)
        )
        actual_dist = mem2_newton_solver(
            moments, guess, direction_increment, twiddle_factors
        )

        actual = 1 - numpy.sum(actual_dist * direction_increment)
        assert_allclose(actual, 0.0, rtol=1e-3, atol=1e-2)

        for m in range(0, 4):
            actual = numpy.sum(
                twiddle_factors[m, :] * actual_dist * direction_increment
            )
            msg = f"moment {m} for case {icase}"
            assert_allclose(actual, moments[m], rtol=1e-3, atol=1e-2, err_msg=msg)


def test_jacobian():
    for icase in range(0, 4):
        moments, _, direction_increment, twiddle_factors = get_case(icase)
        a1, b1, a2, b2 = moments
        lambdas = initial_value(
            numpy.array(a1), numpy.array(b1), numpy.array(a2), numpy.array(b2)
        )
        jacobian = numpy.zeros((4, 4))
        jacobian = mem2_jacobian(
            lambdas, twiddle_factors, direction_increment, jacobian
        )

        def F(x):
            return moment_constraints(x, twiddle_factors, moments, direction_increment)

        for mm in range(0, 4):
            for nn in range(0, 4):
                delta = numpy.zeros(4)
                delta[nn] = 1e-4
                der = (F(lambdas + delta) - F(lambdas - delta)) / delta[nn] / 2
                assert_allclose(jacobian[mm, nn], der[mm])


if __name__ == "__main__":
    test_mem2()
    test_jacobian()
    test_estimate_distribution_newton()
