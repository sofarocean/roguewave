"""
Contents: Routines to calculate (inverse) linear dispersion relation and some related quantities such as phase and
group velocity. NOTE: the effect of surface currents is currently not included in these calculations.

The implementation uses numba to speed up calculations. Consequently, all functions are compiled to machine code, but
the first call to a function will be slow. Subsequent calls will be much faster.

Copyright (C) 2023
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Functions:
- `intrinsic_dispersion_relation`, calculate angular frequency for a given wavenumber and depth
- `inverse_intrinsic_dispersion_relation`, calculate wavenumber for a given angular frequency and depth
- `intrinsic_group_velocity`, calculate the group velocity given wave number and depth
- `phase_velocity`, calculate the phase velocity given wave number and depth
- `ratio_of_group_to_phase_velocity`, calculate the ratio of group to phase velocity given wave number and depth
- `jacobian_wavenumber_to_radial_frequency`, calculate the Jacobian of the wavenumber to radial frequency transformation
- `jacobian_radial_frequency_to_wavenumber`, calculate the Jacobian of the radial frequency to wavenumber transformation
"""

import numpy as np
from numba import jit
from ._tools import atleast_1d
from .constants import GRAV
from ._numba_settings import numba_default
from numbers import Real
from typing import Union


@jit(**numba_default)
def inverse_intrinsic_dispersion_relation(
        angular_frequency: Union[Real,np.ndarray],
        dep: Union[Real,np.ndarray],
        grav:Real=GRAV,
        maximum_number_of_iterations:int=10,
        tolerance:Real=1e-3,
) -> np.ndarray:
    """
    Find wavenumber k for a given radial frequency w using Newton Iteration.
    Exit when either maximum number of iterations is reached, or tolerance
    is achieved. Typically only 1 to 2 iterations are needed.

    :param w: radial frequency
    :param dep: depth in meters
    :param grav:  gravitational acceleration
    :param maximum_number_of_iterations: maximum number of iterations
    :param tolerance: relative accuracy
    :return: The wavenumber as a numpy array.
    """

    # Numba does not recognize "atleast_1d" for scalars
    w = atleast_1d(angular_frequency)

    k_deep_water_estimate = w ** 2 / grav
    k_shallow_water_estimate = w / np.sqrt(grav * dep)

    # == FIRST GUESS==
    # Use the intersection between shallow and deep water estimates to guestimate
    # which relation to use
    wavenumber_estimate = np.where(
        w > np.sqrt(grav / dep), k_deep_water_estimate, k_shallow_water_estimate
    )

    # == Newton Iteration ==
    error = intrinsic_dispersion_relation(wavenumber_estimate, dep, grav) - w
    for ii in range(0, maximum_number_of_iterations):
        # Calculate the derivative of the error function with respect to wavenumber. To note: the derivative is
        # merely the group velocity.
        kd = wavenumber_estimate * dep
        error_derivative_to_wavenumber = np.where(
            kd > 5,
            0.5 * w / wavenumber_estimate,
            (1 / 2 + kd / np.sinh(2 * kd)) * w / wavenumber_estimate
        )
        # Newton Iteration
        wavenumber_estimate = wavenumber_estimate - error / error_derivative_to_wavenumber

        # Update error
        error = intrinsic_dispersion_relation(wavenumber_estimate, dep, grav) - w

        # Check for convergence
        relative_absolute_error = np.abs(error) / w
        if np.all(relative_absolute_error < tolerance):
            break
    else:
        print('inverse_intrinsic_dispersion_relation:: No convergence in solving for wavenumber')

    return wavenumber_estimate


@jit(**numba_default)
def intrinsic_dispersion_relation(k, dep, grav=GRAV) -> np.ndarray:
    """
    The intrinsic dispersion relation for linear waves in water of constant depth that relates the specific angular
    frequency to a given wavenumber and depth in a reference frame following mean ambient flow.

    Wavenumber may be a scalar or a numpy array. The function always returns a numpy array. If depth is specified as a
    numpy array it must have the same shape as the wavenumber array.

    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return: Intrinsic angular frequency (rad/s)
    """
    k = atleast_1d(k)
    return np.sqrt(grav * k * np.tanh(k * dep))


@jit(**numba_default)
def phase_velocity(k, depth, grav=GRAV) -> np.ndarray:
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return intrinsic_dispersion_relation(k, depth, grav=grav) / k


@jit(**numba_default)
def ratio_group_velocity_to_phase_velocity(k, depth, grav) -> np.ndarray:
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    kd = k * depth
    return np.where(kd > 5, 0.5, 0.5 + kd / np.sinh(2 * kd))


@jit(**numba_default)
def intrinsic_group_velocity(k, depth, grav=GRAV) -> np.ndarray:
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return ratio_group_velocity_to_phase_velocity(k, depth, grav=grav) * phase_velocity(
        k, depth, grav
    )


@jit(**numba_default)
def jacobian_wavenumber_to_radial_frequency(k, depth, grav=GRAV) -> np.ndarray:
    """
    :param k: Wavenumber (rad/m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    return 1 / intrinsic_group_velocity(k, depth, grav)


@jit(**numba_default)
def jacobian_radial_frequency_to_wavenumber(k, depth, grav=GRAV) -> np.ndarray:
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
