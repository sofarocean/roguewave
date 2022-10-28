from numpy.typing import ArrayLike
from xarray import DataArray
from numpy import inf, isfinite, abs, nansum, nanmax, sign
import xarray
import numpy
from typing import TypeVar, Callable, List
from roguewave.log import logger
from logging import DEBUG, INFO
from dataclasses import dataclass

_T = TypeVar("_T", ArrayLike, DataArray)

_iteration_depth = 0


@dataclass()
class Configuration:
    atol: float = 1e-4
    rtol: float = 1e-4
    max_iter: int = 100
    aitken_acceleration: bool = True
    fraction_of_points = 1
    error_if_not_converged = False
    numerical_derivative_stepsize = 1e-4


def _log(msg, level):
    if _iteration_depth > 1:
        # Sub iterations are set to level debug
        level = DEBUG
    logger.log(level, msg)


def fixed_point_iteration(
    function: Callable[[_T], _T],
    guess: _T,
    bounds=(-inf, inf),
    caller: str = None,
    configuration: Configuration = None,
) -> _T:
    iterates = [guess, guess, guess]

    if configuration is None:
        configuration = Configuration()

    def _func(iterates):
        return function(iterates[2])

    return _fixed_point_iteration(_func, iterates, bounds, configuration, caller)


def _fixed_point_iteration(
    function: Callable[[List[_T]], _T],
    iterates: List[_T],
    bounds=(-inf, inf),
    configuration: Configuration = None,
    caller: str = None,
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

    # Get the current identation level. This is only used for displaying messages in logs.
    global _iteration_depth
    whitespace = _iteration_depth * "\t"
    # Increase identationm level in case of any recursive calling of solvers.
    _iteration_depth += 1

    if configuration is None:
        configuration = Configuration()

    guess = iterates[2]
    mask_finite_guess_points = numpy.isfinite(guess)
    number_of_active_points = nansum(mask_finite_guess_points)

    is_dataarray = isinstance(guess, DataArray)
    if is_dataarray:
        where = xarray.where
    else:
        where = numpy.where

    if caller is None:
        caller = "unknown"

    msg = f"{whitespace}Starting solver"
    _log(msg, INFO)

    converged = numpy.zeros(guess.shape, dtype="bool")
    for current_iteration in range(1, configuration.max_iter + 1):
        # Update iterate
        aitken_step = configuration.aitken_acceleration and current_iteration % 3 == 0
        if aitken_step:
            # Every third step do an aitken acceleration step- if requested
            ratio = (iterates[2] - iterates[1]) / (iterates[1] - iterates[0])
            ratio = where(numpy.isfinite(ratio) & (ratio != 1.0), ratio, 0.0)

            next_iterate = iterates[2] + ratio / (1.0 - ratio) * (
                iterates[2] - iterates[1]
            )
        else:
            next_iterate = function(iterates)

        # Bounds check
        if isfinite(bounds[0]):
            next_iterate = where(
                (next_iterate <= bounds[0]),
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
        absolute_difference = abs(iterates[2] - iterates[1])

        scale = numpy.maximum(abs(iterates[1]), configuration.atol)
        relative_difference = absolute_difference / scale
        converged[:] = (absolute_difference < configuration.atol) & (
            relative_difference < configuration.rtol
        )

        max_abs_error = nanmax(absolute_difference)
        max_rel_error = nanmax(relative_difference)
        percent_converged = nansum(converged) / number_of_active_points * 100

        number_of_points_converged = nansum(converged)

        msg = f"{whitespace} - Iteration {current_iteration} (max {configuration.max_iter}), convergence in {percent_converged:.2f} % points"
        msg = (
            msg
            + f"max abs. errors: {max_abs_error:.2e} (atol: {configuration.atol}), max rel. error: {max_rel_error:.2e} (rtol: {configuration.rtol})"
        )
        if number_of_points_converged == number_of_active_points and not aitken_step:
            _log(msg, INFO)
            break

        elif (
            number_of_points_converged
            >= number_of_active_points * configuration.fraction_of_points
        ) and not aitken_step:
            _log(msg, INFO)
            break

        else:
            _log(msg, INFO)

    else:
        msg = f"{whitespace}No convergence after {configuration.max_iter}"
        if configuration.error_if_not_converged:
            raise ValueError(msg)

        else:
            iterates[2][~converged] = numpy.nan
            _log(msg, INFO)

    # Reduce identation level in logging
    _iteration_depth -= 1
    return iterates[2]


def secant(
    function: Callable[[_T], _T],
    guess: _T,
    bounds=(-inf, inf),
    configuration: Configuration = None,
    caller: str = None,
) -> _T:
    if configuration is None:
        configuration = Configuration()

    is_dataarray = isinstance(guess, DataArray)
    if is_dataarray:
        where = xarray.where
    else:
        where = numpy.where

    iterates = [guess, guess + configuration.numerical_derivative_stepsize, guess]
    func_evals = [(iterates[1], function(iterates[1]))]

    def _func(iterates):
        prev_val, prev_func = func_evals.pop()

        # Ensure the iterate and the prev func are in sync. During an Aitken update they may go out of sync
        if prev_val is not iterates[1]:
            prev_func = function(iterates[1])

        cur_func = function(iterates[2])
        func_evals.append((iterates[2], cur_func))
        derivative = (cur_func - prev_func) / (iterates[2] - iterates[1])

        update = -cur_func / derivative
        update = where(derivative == 0.0, 0, update)
        update = where(abs(iterates[2] - iterates[1]) < 1e-12, 0, update)
        update = where(
            abs(update) > abs(iterates[2]), sign(update) * abs(iterates[2]), update
        )
        return iterates[2] + update

        # return iterates[2] - cur_func / derivative

    return _fixed_point_iteration(_func, iterates, bounds, configuration, caller)


def newton_raphson(
    function: Callable[[_T], _T],
    guess: _T,
    derivative_function: Callable[[_T], _T] = None,
    bounds=(-inf, inf),
    configuration: Configuration = None,
    caller: str = None,
) -> _T:

    is_dataarray = isinstance(guess, DataArray)
    if is_dataarray:
        where = xarray.where
    else:
        where = numpy.where

    if configuration is None:
        configuration = Configuration()

    iterates = [guess, guess, guess]

    def _func(iterates):
        cur_func = function(iterates[2])

        if derivative_function is None:
            dx = configuration.numerical_derivative_stepsize
            derivative = (function(iterates[2] + dx) - cur_func) / dx
        else:
            derivative = derivative_function(iterates[2])

        update = -cur_func / derivative
        update = where(derivative == 0.0, 0, update)
        update = where(
            abs(update) > abs(iterates[2]), sign(update) * abs(iterates[2]), update
        )
        return iterates[2] + update
        # return where(derivative==0.0,iterates[2],iterates[2] - cur_func / derivative)

    return _fixed_point_iteration(_func, iterates, bounds, configuration, caller)
