"""
Contents: Routines to generate a (monotone) cubic spline interpolation for 1D arrays.

Copyright (C) 2023
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Functions:

- `cubic_spline`, method to create a (monotone) cubic spline

"""
import numpy as np
from scipy.interpolate import CubicSpline

try:
    from qpsolvers import solve_ls
except ImportError:
    pass

def cubic_spline(
    x: np.ndarray,
    y: np.ndarray,
    monotone_interpolation: bool = False,
    frequency_axis=-1,
) -> CubicSpline:
    """
    Construct a cubic spline, optionally monotone.

    :param x: array_like, shape (n,)
              1-D array containing values of the independent variable.
              Values must be real, finite and in strictly increasing order.
    :param y: array_like, shape (...,n)
              set of m 1-D arrays containing values of the dependent variable. Y can have an arbitrary set of leading
              dimensions, but the last dimension has the be equal in size to X. Values must be real, finite and in
              strictly increasing order along the last dimension. (Y is assumed monotone).

    :param monotone_interpolation:
    :return:
    """

    if not monotone_interpolation:
        spline = CubicSpline(x, y, axis=frequency_axis)

    else:
        # Reshape Y, which can have arbitrary leading dimensions (including none) - into a (m,n) shape.
        input_shape = y.shape
        input_axis = list(range(len(input_shape)))
        frequency_axis = input_axis[frequency_axis]

        if len(input_shape) == 1:
            shape = (1, len(x))
            Y = y
        else:
            shape = (
                np.prod(input_shape) // input_shape[frequency_axis],
                input_shape[frequency_axis],
            )
            axis = [axis for axis in input_axis if axis != frequency_axis]
            axis.append(frequency_axis)
            Y = np.transpose(y, axis)
            permuted_input_shape = Y.shape
        Y = np.reshape(Y, shape)

        # Create the spline coeficients
        output = monotone_cubic_spline_coeficients(x, Y)

        # Reshape output so that the aribrary leading dimensions of y become _trailing_ dimensions of the output
        # (this is how spline coeficients are stored in the scipy CubicSpline object)
        if len(input_shape) == 1:
            output_shape = (4, input_shape[-1] - 1)
            output = output.reshape(output_shape)
        else:
            output_shape = (
                *permuted_input_shape[0:-1],
                4,
                permuted_input_shape[-1] - 1,
            )
            output = output.reshape(output_shape)
            output_axis = list(range(len(output_shape)))
            output_axis = output_axis[-2:] + output_axis[:-2]
            output = np.transpose(output, output_axis)

        # Return a CubicSpline object.
        spline = CubicSpline.construct_fast(output, x, extrapolate=False, axis=1)

    return spline


def monotone_cubic_spline_coeficients(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Construct the spline coeficients.
    :param x: array_like, shape (n,)
              1-D array containing values of the independent variable.
              Values must be real, finite and in strictly increasing order.
    :param Y: array_like, shape (m,n)
              set of m 1-D arrays containing values of the dependent variable. For each of the m rows an independent
              spline will be constructed. Values must be real, finite and in strictly increasing order. (Y is assumed
              monotone).
    :param monotone:
    :return:
    """

    output = np.zeros((Y.shape[0], 4, Y.shape[1] - 1))

    # loop over the dimensions
    for jj in range(0, Y.shape[0]):
        _ = _monotone_cubic_spline(x, Y[jj, :], output[jj, :, :])

    return output


def _monotone_cubic_spline(
    x: np.ndarray, y: np.ndarray, spline_coeficients=None
) -> np.ndarray:
    """
    Find a monotone cubic spline function that is maximally smooth. The basic idea is that instead of the traditional
    natural spline - which has C2 continuity but is not guaranteed monotone we reformulate the spline problem as a
    constrained minimization, where the requirements of C0 and C1 continuity and the monoticity conditions form the
    (in)equality constraints, and we try to find a curve that meets these constraints, but is otherwise as close to
    C2 continuity as possible. The spline boundary conditions are not-a-knot boundary conditions, which are added to
    the minimization problem (not as a constraint).

        Wolberg, G., & Alfy, I. (1999, June). Monotonic cubic spline interpolation.
        In Computer Graphics International (pp. 188-195).

        minimize ::

                (matrix @ X - rhs) @ (matrix @ X - rhs)^T

        such that

                equality_constraint_matrix @ X = equality_constraint_rhs
                inequality_constraint_matrix @ X <= inequality_constraint_rhs

    The matrix in the least-squares minimization is formed by minimizing the discontinuity in the second-derivative at
    the spline edges. Note: that because our solver (quadprog) cannot handle matrices that are rank-deficient, we add
    the equality constraints to the minimization problem. This does not alter the result- but allows the solver to
    proceed. (changing solver in future is desirable)

    The equality constraints are formed from the spline conditions that the curve and its first derivative are
    continuous, while we try to minimize the discontinuity in the second derivative at spline edges.

    The inequality constraints are formed from the monoticity conditions (see Wolfberg, 1999).

    :param x: array_like, shape (n,)
              1-D array containing values of the independent variable.
              Values must be real, finite and in strictly increasing order.
    :param y: array_like, shape (n,)
              1-D array containing values of the dependent variable. For each of the m rows an independent
              spline will be constructed. Values must be real, finite and in strictly increasing order. (y is assumed
              monotone).
    :param spline_coeficients: (optional) array_like, shape (4,n)
              Output array to change values in place.
    :return: array_like, shape (4, (n-1))
    """

    # Initialize arrays
    # -----------------
    if spline_coeficients is None:
        spline_coeficients = np.zeros((4, len(x) - 1))

    number_of_splines = x.shape[-1] - 1
    least_squares_matrix = np.zeros((3 * (number_of_splines), 3 * number_of_splines))
    least_squares_rhs = np.zeros((3 * (number_of_splines),))

    equality_constraint_matrix = np.zeros(
        (2 * (number_of_splines - 1) + 1, 3 * number_of_splines)
    )
    equality_constraint_rhs = np.zeros(2 * (number_of_splines - 1) + 1)

    inequality_constraint_matrix = np.zeros(
        (4 * (number_of_splines), 3 * number_of_splines)
    )
    inequality_constraint_rhs = np.zeros((4 * (number_of_splines)))

    delta_x = np.diff(x)
    delta_x = np.reshape(delta_x, (len(delta_x), 1))

    delta_y = np.diff(y)
    secant = np.diff(y) / np.diff(x)
    max_curvature = np.max(
        np.abs(2 * np.diff(y, n=2) / (np.diff(x)[0:-1] + np.diff(x)[1:]))
    )  # 2 * np.diff( y,n=2 ) / (np.diff(x)[0:-1] + np.diff(x)[1:])
    max_secant = np.max(
        np.abs(secant)
    )  # [ np.max( (secant[ii],secant[ii+1]) ) for ii in range(0,len(secant)-1) ]
    max_delta = np.max(np.abs(delta_y))

    if max_delta == 0:
        # Zero solution
        return spline_coeficients

    ones = np.ones_like(delta_x)
    zeros = np.zeros_like(delta_x)
    twos = np.full_like(delta_x, 2)

    # these are the coeficients that represent the linear equations for C0, C1 and C2 continuity at spline edges.
    d2y_dx2_coef = np.concatenate((6 * delta_x, twos, zeros, zeros, -twos), axis=1)
    dy_dx_coef = np.concatenate(
        (3 * delta_x**2, 2 * delta_x, ones, zeros, zeros, -ones), axis=1
    )
    y_coef = np.concatenate((delta_x**3, delta_x**2, delta_x), axis=1)

    # Create Matrices
    # -----------------

    # Not-a-knot boundary condition.
    least_squares_matrix[0, 0] = 1
    least_squares_matrix[0, 3] = -1
    least_squares_matrix[-1, number_of_splines * 3 - 6] = 1
    least_squares_matrix[-1, number_of_splines * 3 - 3] = -1

    # loop over each of the splines, and add its equations
    weights = [1, 10000, 10000]
    for ii in range(0, number_of_splines - 1):
        # row/column indices in the matrix. The unknown spline coeficients are stored as a linear vector of the form
        # solution = [ a0,b0,c0; a1,b1,c1; .... ]. So the spline coeficients of the ii'th spline start in the
        # ii*3'th column
        jcol = ii * 3
        jrow = ii * 3 + 1

        # d2y/dx2 continuity at spline edges. Note technically this is the only equation we minimize
        least_squares_matrix[jrow, jcol : jcol + 5] = (
            d2y_dx2_coef[ii, :] / max_curvature * weights[0]
        )

        # dy/dx continuity at spline edges
        least_squares_matrix[jrow + 1, jcol : jcol + 6] = (
            dy_dx_coef[ii, :] / max_secant * weights[1]
        )

        # y continuity at spline edges
        least_squares_matrix[jrow + 2, jcol : jcol + 3] = (
            y_coef[ii, :] / max_delta * weights[2]
        )
        least_squares_rhs[jrow + 2] = delta_y[ii] / max_delta * weights[2]

        # build the matrix with the equality constrains
        jrow = ii * 2

        # # dy/dx continuity at spline edges.
        if secant[ii] * secant[ii + 1] > 0.0:
            # Note we skip this constraint if we are at a peak in the data- or if one of the surrounding secants is 0.
            # Enforcing 0.0 slope (the other posibility in this case) can lead to a system of equations that is not
            # solvable.
            equality_constraint_matrix[jrow, jcol : jcol + 6] = dy_dx_coef[ii, :]
        else:
            equality_constraint_matrix[jrow, jcol : jcol + 3] = dy_dx_coef[ii, :3]

        equality_constraint_matrix[jrow + 1, jcol : jcol + 3] = y_coef[ii, :]
        equality_constraint_rhs[jrow + 1] = delta_y[ii]

    # Add the end node values for the final spline
    equality_constraint_matrix[-1, -3:] = y_coef[-1, :]
    equality_constraint_rhs[-1] = delta_y[-1]

    least_squares_matrix[-2, number_of_splines * 3 - 3 :] = y_coef[-1, :] / max_delta
    least_squares_rhs[-2] = delta_y[-1] / max_delta

    # Build the matrix for the inequality constraints. These constraints enfore monotone behaviour of each spline.
    eps = 0.0  # 1e-10
    for ii in range(0, number_of_splines):
        jcol = ii * 3
        jrow = ii * 4

        if ii > 0:
            check = secant[ii] * secant[ii - 1] > 0
        else:
            check = np.abs(secant[ii]) > 0

        sign = 1.0 if secant[ii] >= 0 else -1.0

        if check:
            # dy/dx start of spline must be smaller than 3 times the secant
            inequality_constraint_matrix[jrow, jcol + 2] = sign * 1.0
            inequality_constraint_rhs[jrow] = 3 * secant[ii]

            # dy/dx start of spline must be of the same sign as the secant
            inequality_constraint_matrix[jrow + 2, jcol + 2] = -sign
            inequality_constraint_rhs[jrow + 2] = eps
        else:
            # secants change sign across node - slope must be zero-> slope <= 0  & -slope <=0
            inequality_constraint_matrix[jrow, jcol + 2] = 1.0
            inequality_constraint_rhs[jrow] = eps

            inequality_constraint_matrix[jrow + 2, jcol + 2] = -1
            inequality_constraint_rhs[jrow + 2] = eps

        if ii < number_of_splines - 1:
            check = secant[ii + 1] * secant[ii] > 0
        else:
            check = np.abs(secant[ii]) > 0

        if check:
            # dy/dx at the end of the spline must be smaller than 3 times the secant
            inequality_constraint_matrix[jrow + 1, jcol : jcol + 3] = (
                sign * dy_dx_coef[ii, :3]
            )
            inequality_constraint_rhs[jrow + 1] = 3 * secant[ii]

            # dy/dx at the end of the spline must be larger than 0
            inequality_constraint_matrix[jrow + 3, jcol : jcol + 3] = (
                -sign * dy_dx_coef[ii, :3]
            )
            inequality_constraint_rhs[jrow + 3] = eps
        else:
            # secants change sign across node - slope must be zero-> slope <= 0  & -slope <=0
            inequality_constraint_matrix[jrow + 1, jcol : jcol + 3] = dy_dx_coef[ii, :3]
            inequality_constraint_rhs[jrow + 1] = eps

            inequality_constraint_matrix[jrow + 3, jcol : jcol + 3] = -dy_dx_coef[
                ii, :3
            ]
            inequality_constraint_rhs[jrow + 3] = eps

    # Solve system and return results
    # -----------------

    # Solve the solution as a constrained least squares minimization
    solution = solve_ls(
        R=least_squares_matrix,
        s=least_squares_rhs,
        G=inequality_constraint_matrix,
        h=inequality_constraint_rhs,
        A=equality_constraint_matrix,
        b=equality_constraint_rhs,
        verbose=False,
        solver="cvxopt",
    )

    # The unknown spline coeficients are stored as a linear vector of the form:
    #
    #    solution = [ a0,b0,c0; a1,b1,c1; .... ].
    #
    # here we unpack that.
    if solution is None:
        # Remove the dydx continuity requirement.. as a last resort
        equality_constraint_rhs = equality_constraint_rhs[1::2]
        equality_constraint_matrix = equality_constraint_matrix[1::2, :]
        solution = solve_ls(
            R=least_squares_matrix,
            s=least_squares_rhs,
            G=inequality_constraint_matrix,
            h=inequality_constraint_rhs,
            A=equality_constraint_matrix,
            b=equality_constraint_rhs,
            verbose=False,
            solver="cvxopt",
        )
        print(
            "warning - no monotone solution found attempting to solve without C1 constraint"
        )
        if solution is None:
            print("warning - no monotone solution found")
            raise Exception("No solution found.")

    solution = np.reshape(solution, (number_of_splines, 3))

    spline_coeficients[0, :] = solution[:, 0]  # "a" coef
    spline_coeficients[1, :] = solution[:, 1]  # "b" coef
    spline_coeficients[2, :] = solution[:, 2]  # "c" coef
    spline_coeficients[3, :] = y[:-1]  # "d

    #  return the spline coeficients
    return spline_coeficients
