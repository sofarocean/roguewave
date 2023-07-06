import numpy
from numba import njit, types
from numba.extending import overload

GRAV = 9.81

# The following overloading trick is needed because "atleast_1d" is not supported for scalars by default in numba.
def atleast_1d(x) -> numpy.ndarray:
    if type(x) in types.number_domain:
        return numpy.array([x])
    return numpy.atleast_1d(x)

@overload(atleast_1d)
def overloaded_atleast_1d(x):
    if x in types.number_domain:
        return lambda x: numpy.array([x])
    return lambda x: numpy.atleast_1d(x)

def atleast_2d(x) -> numpy.ndarray:
    if x in types.number_domain:
        return numpy.array([x])
    return numpy.atleast_1d(x)

@overload(atleast_2d)
def overloaded_atleast_2d(x):
    if x in types.number_domain:
        return lambda x: numpy.array([[x]])
    return lambda x: numpy.atleast_2d(x)

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

    # Numba does not recognize "atleast_1d" for scalars
    w = atleast_1d(angular_frequency)

    k_deep_water_estimate = w**2 / grav
    k_shallow_water_estimate = w / numpy.sqrt(grav * dep)

    # == FIRST GUESS==
    # Use the intersection between shallow and deep water estimates to guestimate
    # which relation to use
    k0 = numpy.where(
        w > numpy.sqrt(grav / dep), k_deep_water_estimate, k_shallow_water_estimate
    )

    # == Newton Iteration ==
    F = numpy.sqrt(k0 * grav * numpy.tanh(k0 * dep)) - w
    for ii in range(0, maximum_number_of_iterations):
        kd = k0 * dep
        cg = numpy.where(
            kd > 5, 0.5 * w / k0, (1 / 2 + kd / numpy.sinh(2 * kd)) * w / k0
        )
        k0 = k0 - F / cg
        F = numpy.sqrt(k0 * grav * numpy.tanh(k0 * dep)) - w
        error = numpy.abs(F) / w
        if numpy.all(error < tolerance):
            break
    else:
        print('inverse_intrinsic_dispersion_relation:: No convergence in solving for wavenumber')

    return k0


@njit(cache=True)
def intrinsic_dispersion_relation(k, dep, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    k = atleast_1d(k)
    return numpy.sqrt(grav * k * numpy.tanh(k * dep))


@njit(cache=True)
def phase_velocity(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return intrinsic_dispersion_relation(k, depth, grav=grav) / k


@njit(cache=True)
def ratio_group_velocity_to_phase_velocity(k, depth, grav):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    kd = k * depth
    return numpy.where(kd > 5, 0.5, 0.5 + kd / numpy.sinh(2 * kd))


@njit(cache=True)
def intrinsic_group_velocity(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return ratio_group_velocity_to_phase_velocity(k, depth, grav=grav) * phase_velocity(
        k, depth, grav
    )


@njit(cache=True)
def jacobian_wavenumber_to_radial_frequency(k, depth, grav=GRAV):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
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

@njit(cache=True)
def dispersion_relation( kx,ky=None,ux=None,uy=None, depth=numpy.inf, grav=GRAV ):
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param current: Representative current (m/s)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    kx = atleast_1d(kx)
    ux = numpy.zeros_like(kx) if ux is None else atleast_1d(ux)
    uy = numpy.zeros_like(kx) if uy is None else atleast_1d(uy)
    ky = numpy.zeros_like(kx) if uy is None else atleast_1d(ky)

    k = numpy.sqrt(kx**2 + ky**2)
    doppler_shift = kx*ux + ky*uy
    return intrinsic_dispersion_relation(k, depth,grav) + doppler_shift

def inverse_dispersion_relation( omega, direction=None, ux=None,uy=None, depth=numpy.inf, grav=GRAV):
    pass

# Aliasses based on common notation in linear wave theory
c = phase_velocity
cg = intrinsic_group_velocity
k = inverse_intrinsic_dispersion_relation
w = intrinsic_dispersion_relation
n = ratio_group_velocity_to_phase_velocity
