import numpy
from roguewave.tools.math import wrapped_difference
from roguewave.tools.grid import enclosing_points_1d


def interpolate_periodic(
    xp: numpy.ndarray,
    fp: numpy.ndarray,
    x: numpy.ndarray,
    x_period: float = None,
    fp_period: float = None,
    fp_discont: float = None,
    left: float = numpy.nan,
    right: float = numpy.nan,
) -> numpy.ndarray:
    """
    Interpolation function that works with periodic domains and periodic ranges
    :param xp:
    :param fp:
    :param x:
    :param x_period:
    :param fp_period:
    :param fp_discont:
    :return:
    """

    indices = enclosing_points_1d(xp, x, period=x_period, regular_xp=False)

    delta_x = wrapped_difference(x - xp[indices[0, :]], period=x_period)
    delta_xp = wrapped_difference(
        xp[indices[1, :]] - xp[indices[0, :]], period=x_period
    )
    delta_fp = wrapped_difference(
        delta=fp[indices[1, :]] - fp[indices[0, :]], period=fp_period
    )
    fp0 = fp[indices[0, :]]

    if x_period is not None:
        interpolation = fp0 + delta_fp * delta_x / delta_xp

    else:
        # constant extrapolation
        mask = delta_xp.astype("float64") > 0.0
        interpolation = numpy.zeros((len(x),))
        interpolation[mask] = (
            fp0[mask] + delta_fp[mask] * delta_x[mask] / delta_xp[mask]
        )
        interpolation[~mask] = fp0[~mask]

        # substitute value if given
        if left is not None:
            interpolation = numpy.where(x < xp[0], left, interpolation)

        # substitute value if given
        if right is not None:
            interpolation = numpy.where(x > xp[-1], right, interpolation)

    mask = numpy.isfinite(interpolation)
    interpolation[mask] = wrapped_difference(
        delta=interpolation[mask], period=fp_period, discont=fp_discont
    )

    return interpolation


def interpolation_weights_1d(
    xp, x, indices, period: float = None, extrapolate_left=True, extrapolate_right=True
):
    """
    Find the weights for the linear interpolation problem given a set of
    indices such that:

                xp[indices[0,:]] <= x[:] < xp[indices[1,:]]

    Indices are assumed to be as returned by "enclosing_points_1d" (see
    roguewave.tools.grid).

    Returns weights (nx,2) to perform the one-dimensional piecewise linear
    interpolation to a function given at discrete datapoints xp and evaluated
    at x. Say that at all xp we have for the function values fp,
    the interpolation would then be

        f = fp[ Indices[1,:] ]  *  weights[1,:] +
            fp[ Indices[2,:] ]  *  weights[2,:]

    :param xp:
    :param x:
    :param indices:
    :param period:
    :return:
    """

    weights = numpy.empty_like(indices, dtype="float64")

    if xp[-1] < xp[0]:
        # make sure we are in a coordinate frame where the vector is
        # in ascending order.
        x = xp[0] - x
        xp = xp[0] - xp

    delta_x = wrapped_difference(x - xp[indices[0, :]], period=period)

    delta_xp = wrapped_difference(xp[indices[1, :]] - xp[indices[0, :]], period=period)

    if period is None:
        mask = (x >= xp[0]) & (x < xp[-1])
        frac = numpy.empty((len(x),))
        frac[mask] = delta_x[mask] / delta_xp[mask]
        if extrapolate_left:
            frac[x < xp[0]] = 1
        else:
            frac[x < xp[0]] = numpy.nan

        if extrapolate_right:
            frac[x > xp[-1]] = 0
        else:
            frac[x > xp[-1]] = numpy.nan

        # At the right end point to avoid devision by 0.
        frac[x == xp[-1]] = 0.0
    else:
        frac = delta_x / delta_xp

    weights[0, :] = 1 - frac
    weights[1, :] = frac

    return weights
