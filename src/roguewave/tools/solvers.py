from numpy.typing import ArrayLike
from xarray import DataArray
from numpy import inf, isfinite, abs, all, sum, max
import xarray
import numpy
from typing import TypeVar, Callable, List
from roguewave.log import logger

_T = TypeVar("_T", ArrayLike, DataArray)


def fixed_point_iteration(
    function: Callable[[_T], _T],
    guess: _T,
    bounds=(-inf, inf),
    max_iter=100,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    caller: str = None,
    aitken_accelerate=True,
    fraction_of_points=1,
    error_if_not_converged=True,
) -> _T:
    iterates = [guess, guess, guess]

    def _func(iterates):
        return function(iterates[2])

    return _fixed_point_iteration(
        _func,
        iterates,
        bounds,
        max_iter,
        atol,
        rtol,
        caller,
        aitken_accelerate,
        fraction_of_points,
        error_if_not_converged,
    )


def _fixed_point_iteration(
    function: Callable[[List[_T]], _T],
    iterates: List[_T],
    bounds=(-inf, inf),
    max_iter=100,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    caller: str = None,
    aitken_accelerate=True,
    fraction_of_points=1,
    error_if_not_converged=True,
) -> _T:
    """
    Fixed point iteration on a vector function. We want to solve the parallal problem x=F(x) where x is a vector. Instead
    of looping over each problem and solving them individualy using e.g. scipy solvers, we gain some efficiency by
    evaluating F in parallel, and doing the iteration ourselves. Only worthwhile if F is the expensive part and/or x
    is large.

    :param function:
    :param guess:
    :param max_iter:
    :param atol:
    :param rtol:
    :param caller:
    :return:
    """
    guess = iterates[2]
    msk = numpy.isfinite(guess)
    is_dataarray = isinstance(guess, DataArray)
    if is_dataarray:
        where = xarray.where
    else:
        where = numpy.where

    if caller is None:
        caller = "unknown"

    msg = "Starting solver"
    logger.info(msg)

    for current_iteration in range(1, max_iter + 1):
        # Update iterate
        aitken_step = aitken_accelerate and current_iteration % 3 == 0
        if aitken_step:
            # Every third step do an aitken acceleration step- if requested
            ratio = (iterates[2] - iterates[1]) / (iterates[1] - iterates[0])
            next_iterate = iterates[2] + ratio / (1 - ratio) * (
                iterates[2] - iterates[1]
            )
        else:
            next_iterate = function(iterates)

        # Bounds check
        if isfinite(bounds[0]):
            next_iterate = where(
                (next_iterate < bounds[0]),
                (bounds[0] - iterates[2]) * 0.5 + iterates[2],
                next_iterate,
            )

        if isfinite(bounds[1]):
            next_iterate = where(
                (next_iterate > bounds[1]),
                (bounds[1] - iterates[2]) * 0.5 + iterates[2],
                next_iterate,
            )

        # Roll the iterates, make the last entry the latest estimate
        iterates.append(iterates.pop(0))
        iterates[2] = next_iterate

        # Convergence check
        absolute_difference = (abs(iterates[2] - iterates[1]))[msk]

        scale = numpy.maximum(abs(iterates[1][msk]), atol)
        relative_difference = absolute_difference / scale
        converged = (absolute_difference < atol) & (relative_difference < rtol)

        max_abs_error = max(absolute_difference)
        max_rel_error = max(relative_difference)
        percent_converged = sum(converged) / len(converged) * 100

        if is_dataarray:
            max_rel_error = max_rel_error.values
            max_abs_error = max_abs_error.values
            percent_converged = percent_converged.values

        msg = f"\tIteration {current_iteration} (max {max_iter}), convergence in {percent_converged:.2f} % points"
        msg = (
            msg
            + f"   max abs. errors: {max_abs_error:.2e} (atol: {atol}), max rel. error: {max_rel_error:.2e} (rtol: {rtol})"
        )
        if all(converged) and not aitken_step:
            logger.info(msg)
            logger.info("Converged \n")
            break

        elif (sum(converged) > len(converged) * fraction_of_points) and not aitken_step:
            logger.info(msg)
            break

        else:
            logger.info(msg)

    else:
        msg = f"No convergence after {max_iter}"
        if error_if_not_converged:
            raise ValueError(msg)

        else:
            logger.info(msg)

    return iterates[2]


def secant(
    function: Callable[[_T], _T],
    guess: _T,
    bounds=(-inf, inf),
    max_iter=100,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    caller: str = None,
    aitken_accelerate=True,
    fraction_of_points=1,
    error_if_not_converged=True,
) -> _T:

    iterates = [guess, guess + 0.00001, guess]
    func_evals = [(iterates[1], function(iterates[1]))]

    def _func(iterates):
        prev_val, prev_func = func_evals.pop()

        # Ensure the iterate and the prev func are in sync. During an Aitken update they may go out of sync
        if prev_val is not iterates[1]:
            prev_func = function(iterates[1])

        cur_func = function(iterates[2])
        derivative = (cur_func - prev_func) / (iterates[2] - iterates[1])
        func_evals.append((iterates[2], cur_func))
        return iterates[2] - cur_func / derivative

    return _fixed_point_iteration(
        _func,
        iterates,
        bounds,
        max_iter,
        atol,
        rtol,
        caller,
        aitken_accelerate,
        fraction_of_points,
        error_if_not_converged,
    )
