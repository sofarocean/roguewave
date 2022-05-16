import numpy
import typing

PHILLIPS_CONSTANT = 0.0081
GRAVITATIONAL_CONSTANT = 9.81

def pierson_moskowitz_frequency(frequency: typing.Union[numpy.ndarray, float],
                                peak_frequency: float, alpha=PHILLIPS_CONSTANT, g=GRAVITATIONAL_CONSTANT):
    """
    Pierson Moskowitz variance-density spectrum with frequency in Hz as
    dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

    :param frequency: frequency in Hz (scalar or array)
    :param peak_frequency: peak frequency in Hz
    :param alpha: Phillips constant (default 0.0081)
    :param g: gravitational acceleration (default 9.81)
    :return:
    """
    return alpha * g ** 2 * (2 * numpy.pi) ** -4 * frequency ** -5 * numpy.exp(
        - 5 / 4 * (peak_frequency / frequency) ** 4)


def pierson_moskowitz_angular_frequency(
        angular_frequency: typing.Union[numpy.ndarray, float],
        peak_angular_frequency: float, alpha=PHILLIPS_CONSTANT, g=GRAVITATIONAL_CONSTANT):
    """
    Pierson Moskowitz variance-density spectrum with angular frequency (rad/s) as
    dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

    :param frequency: frequency in rad/s (scalar or array)
    :param peak_frequency: peak frequency in rad/s
    :param alpha: Phillips constant (default 0.0081)
    :param g: gravitational acceleration (default 9.81)
    :return: 
    """
    return alpha * g ** 2 * angular_frequency ** -5 * numpy.exp(
        - 5 / 4 * (peak_angular_frequency / angular_frequency) ** 4)


def parametric( shape, directional_shape ):
    pass