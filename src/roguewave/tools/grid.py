import numpy
from numba import njit


#@njit(cache=True)
def enclosing_points_1d(
        xp: numpy.ndarray,
        x: numpy.ndarray,
        regular_xp=False,
        period: float = None
):
    """
    Find surrounding indices for value x[j] in vector xp such
    that  xp[i-1] <= x[j] < xp[i]

    To note for non-periodic sequences values outside of xp require special
    attention- specifically:
        for x < xp[0]:
            we have x[x<xp[0]] < xp[indices[1,:]], and indices[0,:] is
            undefined (clipped to 0).

        for x >= xp[-1]:
            we have x[x>=xp[-1]] >= xp[indices[0,:]], and indices[1,:] is
            undefined (clipped to nx).

    Parameters
    ----------
    :param x: 1-D array_like of length nx
            The x-coordinates at which to evaluate the interpolated values with
            nx entries.

    :param xp: 1-D sequence of floats
        The x-coordinates of the data points with nxp entries.

    :return: [2 by nx] numpy array of integer (dtype='int64') indices such that
        xp[indices[0,:]] <= x[:] < xp[indices[1,:]] with the exception of
        points x outside of the domain of xp.


    """

    x = numpy.atleast_1d(x)
    nxp = len(xp)
    nx = len(x)

    if xp[-1] < xp[0]:
        # make sure we are in a coordinate frame where the vector is
        # in ascending order.
        x =  xp[0] - x
        xp = xp[0] - xp

    if period is not None:
        x = (x - xp[0]) % period + xp[0]

    indices = numpy.empty((2, nx), dtype='int64')
    if not regular_xp:
        for ix, x_ix in enumerate(x):
            # find ixp such that  xp[ixp-1] <= x[ix] < xp[ixp]
            ixp = numpy.searchsorted(xp, x_ix, side='right')
            indices[:, ix] = [ixp - 1, ixp]
    else:
        ixp = numpy.floor((x - xp[0]) / (xp[1] - xp[0]))
        indices[0, :] = ixp
        indices[1, :] = ixp + 1

    if period is not None:
        indices = indices % nxp
    else:
        indices = numpy.clip(indices, 0, nxp - 1)
    return indices
