from numpy import inf, abs
from numpy.typing import ArrayLike
from numba import njit
from xarray import DataArray
from typing import TypeVar

_T = TypeVar("_T", ArrayLike, DataArray)


@njit(fastmath=True)
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
