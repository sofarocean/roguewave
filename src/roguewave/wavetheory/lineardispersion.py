import numpy
from numba import njit
GRAV = 9.81

@njit(cache=True)
def inverse_intrinsic_dispersion_relation(
    angular_frequency,
    dep,
    grav=GRAV,
    maximum_number_of_iterations=10,
    tolerance=1e-3,
):
    """
    Find wavenumber k for a given radial frequency w using Newton Iteration.
    Exit when either maximum number of iterations is reached, or tolerance
    is achieved. Typically only 1 to 2 iterations are needed.

    :param w: radial frequency
    :param dep: depth in meters
    :param grav:  gravitational acceleration
    :param maximum_number_of_iterations: maximum number of iterations
    :param tolerance: relative accuracy
    :return:
    """

    # Numba does not recognize "atleast_1d" for scalars - hence the weird
    # call to array first.
    w = numpy.atleast_1d( numpy.array(angular_frequency))

    k_deep_water_estimate = w ** 2 / grav
    k_shallow_water_estimate = w / numpy.sqrt(grav * dep)
    k0 = numpy.zeros(w.shape, dtype=w.dtype)

    # == FIRST GUESS==
    # Use the intersection between shallow and deep water estimates to guestimate
    # which relation to use
    msk = w > numpy.sqrt(grav / dep)
    k0[msk] = k_deep_water_estimate[msk]

    msk = numpy.logical_not(msk)
    k0[msk] = k_shallow_water_estimate[msk]

    # == Newton Iteration ==
    F = numpy.sqrt(k0 * grav * numpy.tanh(k0 * dep)) - w
    cg = numpy.zeros(w.shape, dtype=w.dtype)

    for ii in range(0, maximum_number_of_iterations):
        kd = k0 * dep
        msk = kd > 3
        cg[msk] = 0.5 * w[msk] / k0[msk]
        msk = kd <= 3
        cg[msk] = (1 / 2 + kd[msk] / numpy.sinh(2 * kd[msk])) * w[msk] / k0[msk]
        k0 = k0 - F / cg

        F = numpy.sqrt(k0 * grav * numpy.tanh(k0 * dep)) - w
        error = numpy.abs(F) / w
        if numpy.all(error < tolerance):
            break

    return k0

@njit(cache=True)
def intrinsic_dispersion_relation(k, dep, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    k = numpy.atleast_1d(numpy.array(k))
    w = numpy.sqrt(grav * k * numpy.tanh(k * dep))
    return w

@njit(cache=True)
def phase_velocity(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return intrinsic_dispersion_relation(k, depth, grav=GRAV) / k

@njit(cache=True)
def ratio_group_velocity_to_phase_velocity(k, depth, grav):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    kd = k * depth
    n = numpy.zeros(kd.shape, dtype=k.dtype)

    msk = kd > 3
    n[msk] = 0.5

    msk = kd <= 3
    n[msk] = 0.5 + kd[msk] / numpy.sinh(2 * kd[msk])

    return n

@njit(cache=True)
def intrinsic_group_velocity(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return ratio_group_velocity_to_phase_velocity(k, depth, grav=GRAV) * phase_velocity(k, depth, grav)

@njit(cache=True)
def jacobian_wavenumber_to_radial_frequency(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    # return numpy.ones( k.shape  )
    return 1 / intrinsic_group_velocity(k, depth, grav)

@njit(cache=True)
def jacobian_radial_frequency_to_wavenumber(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return intrinsic_group_velocity(k, depth, grav)


# Aliasses based on common notation in linear wave theory
c = phase_velocity
cg = intrinsic_group_velocity
k = inverse_intrinsic_dispersion_relation
w = intrinsic_dispersion_relation
n = ratio_group_velocity_to_phase_velocity
