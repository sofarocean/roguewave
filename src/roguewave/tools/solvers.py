from numpy.typing import ArrayLike
from xarray import DataArray
from numpy import inf, isfinite, abs, nansum, nanmax, sign
import xarray
import numpy
from typing import TypeVar, Callable, List
from roguewave.log import logger
from logging import DEBUG, INFO
from dataclasses import dataclass
from numba import njit

_T = TypeVar("_T", ArrayLike, DataArray)

_iteration_depth = 0


@dataclass()
class Configuration:
    atol: float = 1e-4
    rtol: float = 1e-4
    max_iter: int = 100
    aitken_acceleration: bool = True
    fraction_of_points: float = 1
    error_if_not_converged: bool = False
    numerical_derivative_stepsize: float = 1e-4
    use_numba: bool = True


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


@njit()
def numba_fixed_point_iteration(
    function,
    guess,
    args,
    bounds=(-inf, inf),
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

    iterates = [guess, guess, guess]
    max_iter = 100
    aitken_acceleration = True
    rtol = 1e-4
    atol = 1e-4
    for current_iteration in range(1, max_iter + 1):
        # Update iterate
        aitken_step = aitken_acceleration and current_iteration % 3 == 0
        if aitken_step:
            # Every third step do an aitken acceleration step- if requested
            numerator = iterates[2] - iterates[1]
            denominator = iterates[1] - iterates[0]

            if denominator != 0.0 and numerator != denominator:
                ratio = numerator / denominator
                next_iterate = iterates[2] + ratio / (1.0 - ratio) * (numerator)
            else:
                next_iterate = iterates[2]
        else:
            next_iterate = function(iterates[2], *args)

        # Bounds check
        if next_iterate < bounds[0]:
            next_iterate = (bounds[0] - iterates[2]) * 0.5 + iterates[2]

        if next_iterate > bounds[1]:
            next_iterate = (bounds[1] - iterates[2]) * 0.5 + iterates[2]

        # Roll the iterates, make the last entry the latest estimate
        iterates.append(iterates.pop(0))
        iterates[2] = next_iterate

        # Convergence check
        scale = max(abs(iterates[1]), atol)
        absolute_difference = abs(iterates[2] - iterates[1])
        relative_difference = absolute_difference / scale

        if (absolute_difference < atol) & (
            relative_difference < rtol
        ) and not aitken_step:
            break
    else:
        raise ValueError("no convergence")

    return iterates[2]


@njit()
def numba_newton_raphson(
    function,
    guess,
    function_arguments,
    hard_bounds=(-inf, inf),
    max_iterations=100,
    aitken_acceleration=True,
    atol=1e-4,
    rtol=1e-4,
    numerical_stepsize=1e-4,
    verbose=False,
):

    # The last three iterates. Newest iterate last. We initially fill all with the guess.
    iterates = [guess, guess, guess]

    # The function evaluatio at the last three iterate, last evation last. We initially fill all with 0. so that entries
    # are invalid until we get to the 3rd iterate.
    func_evals = [0.0, 0.0, 0.0]

    # Guess root bounds. Initially this may not actually bound the root, but if the guess is reasonable, it may.
    root_bounds = [guess - 0.5 * abs(guess), guess + 0.5 * abs(guess)]

    # The function evaluated at the bounds. We may get lucky and immediately bound the root.
    func_at_bounds = [
        function(root_bounds[0], *function_arguments),
        function(root_bounds[1], *function_arguments),
    ]

    # Is the root bounded?
    root_bounded = func_at_bounds[0] * func_at_bounds[1] < 0

    # Start iteration.
    for current_iteration in range(1, max_iterations):

        # Evaluate the function at the latest iterate. First roll the list...
        func_evals.append(func_evals.pop(0))
        # ... and then update the latest point.
        func_evals[2] = function(iterates[2], *function_arguments)

        # Every 3rd step we do a Aitken series acceleration step
        aitken_step = aitken_acceleration and current_iteration % 3 == 0
        if aitken_step:
            # We do an Aitkon step
            numerator = iterates[2] - iterates[1]
            denominator = iterates[1] - iterates[0]
            ratio = numerator / denominator
            next_iterate = iterates[2] + ratio / (1.0 - ratio) * (numerator)

        else:
            # We use a regular update step. This can either be: numerical estimate for Newton Raphson, Secant, or a
            # bisection step.

            # First lets find out if our itereate is within the root bounds. Initially, when the root is not yet bounded
            # we will escape the bounds as we do gradient descent with Newton.
            if iterates[2] < root_bounds[0]:
                # our iterate is smaller than the left bound - set the left bound to the iterate, but respect the overal
                # bounds...
                root_bounds[0] = iterates[2]
                func_at_bounds[0] = func_evals[2]

            elif iterates[2] > root_bounds[1]:
                root_bounds[1] = iterates[2]
                func_at_bounds[1] = func_evals[2]

            # If we are within the current bounds, does any of the sections contain the root?
            elif func_at_bounds[0] * func_evals[2] < 0:

                func_at_bounds[1] = func_evals[2]
                root_bounds[1] = iterates[2]
            elif func_at_bounds[1] * func_evals[2] < 0:
                root_bounded = True
                func_at_bounds[0] = func_evals[2]
                root_bounds[0] = iterates[2]

            # With the boundaries updated, check if we now bound the root.
            root_bounded = func_at_bounds[0] * func_at_bounds[1] < 0

            # If we bound the root we can start using secant - if it wants to escape the bounds we will just use a
            # bisection step, if it doesn't secant  is faster than bisection.
            if root_bounded and current_iteration > 1:
                # Secant estimate
                derivative = (func_evals[2] - func_evals[1]) / (
                    iterates[2] - iterates[1]
                )

            else:
                # Finite difference estimate
                derivative = (
                    function(iterates[2] + numerical_stepsize, *function_arguments)
                    - func_evals[2]
                ) / numerical_stepsize

            # We encountered a stationary point.
            if derivative == 0.0:
                # If we have the root bounded- let's do a bisection step to keep going.
                if root_bounded:
                    update = (root_bounds[1] - root_bounds[0]) / 2

                else:
                    raise ValueError("")

            else:
                update = -func_evals[2] / derivative

            next_iterate = iterates[2] + update

        # Bounds check
        if not root_bounded:
            bounds_to_check = hard_bounds
        else:
            bounds_to_check = (root_bounds[0], root_bounds[1])

        if next_iterate < bounds_to_check[0]:
            next_iterate = (bounds_to_check[0] - iterates[2]) * 0.5 + iterates[2]

        if next_iterate > bounds_to_check[1]:
            next_iterate = (bounds_to_check[1] - iterates[2]) * 0.5 + iterates[2]

        # Roll the iterates, make the last entry the latest estimate
        iterates.append(iterates.pop(0))
        iterates[2] = next_iterate

        # Convergence check
        scale = max(abs(iterates[1]), atol)
        absolute_difference = abs(iterates[2] - iterates[1])
        relative_difference = absolute_difference / scale

        if verbose:
            # In case anybody wonders - this monstrosity is needed because Numba does not currently support converting
            # floats to strings.
            print(
                "Iteration",
                current_iteration,
                "max abs. errors:",
                absolute_difference,
                "(atol:",
                atol,
                "max rel. error:",
                relative_difference,
                "(rtol: ",
                rtol,
                ")",
            )

        if (absolute_difference < atol) & (relative_difference < rtol):
            break

    else:
        raise ValueError("no convergence")

    return iterates[2]


@njit(cache=True, fastmath=True)
def solve_cholesky(matrix, rhs):
    """
    Solve using cholesky decomposition according to the Cholesky–Banachiewicz algorithm.
    See: https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky_algorithm
    """
    M, N = matrix.shape
    x = numpy.zeros(M)
    cholesky_decomposition = numpy.zeros((M, M))
    inv = numpy.zeros(M)

    for mm in range(0, M):
        forward_sub_sum = rhs[mm]
        for nn in range(0, mm):
            sum = matrix[mm, nn]
            for kk in range(0, nn):
                sum -= cholesky_decomposition[mm, kk] * cholesky_decomposition[nn, kk]

            cholesky_decomposition[mm, nn] = inv[nn] * sum
            forward_sub_sum += -cholesky_decomposition[mm, nn] * x[nn]

        sum = matrix[mm, mm]
        for kk in range(0, mm):
            sum -= cholesky_decomposition[mm, kk] ** 2

        if sum <= 0.0:
            raise ValueError(
                "Matrix not positive definite, likely due to finite precision errors."
            )

        cholesky_decomposition[mm, mm] = numpy.sqrt(sum)
        inv[mm] = 1 / cholesky_decomposition[mm, mm]
        x[mm] = forward_sub_sum * inv[mm]

    # Backward Substitution (in place)
    for mm in range(0, M):
        kk = M - mm - 1
        sum = x[kk]
        for nn in range(kk + 1, N):
            sum += -cholesky_decomposition[nn, kk] * x[nn]
        x[kk] = sum * inv[kk]
    return x
