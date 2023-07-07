import numpy as np
from numba import njit
from ._numba_settings import numba_default
from ._tools import atleast_1d
from .lineardispersion import intrinsic_dispersion_relation
from .constants import GRAV, WATER_DENSITY


@njit(**numba_default)
def horizontal_particle_velocity_amplitude(surface_amplitude, k, z, depth, surface_elevation=0, grav=GRAV):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param direction: Direction (rad)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    k = atleast_1d(k)
    z = atleast_1d(z)
    angular_frequency = intrinsic_dispersion_relation(k, depth, grav=grav)

    s = s_coordinate(z, depth, surface_elevation)
    kd = k * depth
    ch = np.cosh(kd * (1 + s)) / np.cosh(kd)

    return surface_amplitude * ch * grav * k / angular_frequency


@njit(**numba_default)
def vertical_particle_velocity_amplitude(surface_amplitude, k, z, depth, surface_elevation=0, grav=GRAV):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param direction: Direction (rad)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    k = atleast_1d(k)
    z = atleast_1d(z)
    angular_frequency = intrinsic_dispersion_relation(k, depth, grav=grav)

    s = s_coordinate(z, depth, surface_elevation)
    kd = k * depth
    sh = np.sinh(kd * (1 + s)) / np.cosh(kd)

    return surface_amplitude * sh * grav * k / angular_frequency


@njit(**numba_default)
def particle_velocity_amplitude_x(surface_amplitude, direction, k, z, depth, surface_elevation=0,
                                  grav=GRAV):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param direction: Direction (rad)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """

    return horizontal_particle_velocity_amplitude(surface_amplitude, k, z, depth, direction=direction,
                                                  surface_elevation=surface_elevation, grav=grav) * np.cos(direction)


@njit(**numba_default)
def particle_velocity_amplitude_y(surface_amplitude, direction, k, z, depth, surface_elevation=0,
                                  grav=GRAV):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param direction: Direction (rad)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """

    return horizontal_particle_velocity_amplitude(surface_amplitude, k, z, depth, direction=direction,
                                                  surface_elevation=surface_elevation, grav=grav) * np.sin(direction)


@njit(**numba_default)
def particle_velocity_amplitude_z(surface_amplitude, k, z, depth, surface_elevation=0, grav=GRAV):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """

    return -1j * vertical_particle_velocity_amplitude(surface_amplitude, k, z, depth,
                                                      surface_elevation=surface_elevation, grav=grav)


def pressure_amplitude(surface_amplitude, k, z, depth, surface_elevation=0, grav=GRAV, density=WATER_DENSITY):
    """
    :param surface_amplitude: Surface amplitude (m)
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    k = atleast_1d(k)
    z = atleast_1d(z)
    s = s_coordinate(z, depth, surface_elevation)
    ch = np.cosh(k * depth * (1 + s)) / np.cosh(k * depth)

    return density * grav * surface_amplitude * (ch - (depth + z) / (depth + surface_elevation))


@njit(**numba_default)
def s_coordinate(z, depth, surface_elevation):
    """
    :param k: Wavenumber (rad/m)
    :param z: Depth (m)
    :param depth: Depth (m)
    :param direction: Direction (rad)
    :param grav: Gravitational acceleration (m/s^2)
    :return:
    """
    z = atleast_1d(z)
    return z - surface_elevation / (depth + surface_elevation)
