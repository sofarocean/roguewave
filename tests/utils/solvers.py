from roguewave.tools.solvers import fixed_point_iteration, secant
import numpy
from numpy.testing import assert_allclose


def fixed_point_test():
    # set_log_to_console()

    def _func(x: numpy.ndarray) -> numpy.ndarray:
        return 6.28 + numpy.sin(x)

    x0 = numpy.zeros(10)
    rtol = 1e-4
    atol = 1e-4

    res1 = fixed_point_iteration(
        _func, x0, rtol=rtol, atol=atol, aitken_accelerate=True, max_iter=200
    )
    res2 = fixed_point_iteration(
        _func, x0, rtol=rtol, atol=atol, aitken_accelerate=False, max_iter=200
    )

    assert_allclose(res1, _func(res1), rtol=rtol, atol=atol)
    assert_allclose(res2, _func(res2), rtol=rtol, atol=atol)


def t_secant():
    # set_log_to_console()

    def _func(x: numpy.ndarray) -> numpy.ndarray:
        return numpy.sin(x)

    x0 = numpy.zeros(1) + 1
    rtol = 1e-4
    atol = 1e-4

    res1 = secant(_func, x0, rtol=rtol, atol=atol, aitken_accelerate=True, max_iter=200)
    res2 = secant(
        _func, x0, rtol=rtol, atol=atol, aitken_accelerate=False, max_iter=200
    )

    assert_allclose(_func(res1), 0, rtol=rtol, atol=atol)
    assert_allclose(_func(res2), 0, rtol=rtol, atol=atol)


# fixed_point_test()
t_secant()
