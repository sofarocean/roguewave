from .spectrum import FrequencyDirectionSpectrum, create_2d_spectrum
from abc import ABC, abstractmethod
from scipy.special import gamma
from datetime import datetime
import numpy
import typing
from typing import Literal

PHILLIPS_CONSTANT = 0.0081
GRAVITATIONAL_CONSTANT = 9.81

FrequencyShapeOptions = Literal["pm"]
DirectionalShapeOptions = Literal["raised_cosine"]


def pierson_moskowitz_frequency(
    frequency, peak_frequency, alpha=PHILLIPS_CONSTANT, g=GRAVITATIONAL_CONSTANT
):
    """
    Pierson Moskowitz variance-density spectrum with frequency in Hz as
    dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

    :param frequency: frequency in Hz (scalar or array)
    :param peak_frequency: peak frequency in Hz
    :param alpha: Phillips constant (default 0.0081)
    :param g: gravitational acceleration (default 9.81)
    :return:
    """
    return (
        alpha
        * g**2
        * (2 * numpy.pi) ** -4
        * frequency**-5
        * numpy.exp(-5 / 4 * (peak_frequency / frequency) ** 4)
    )


def pierson_moskowitz_angular_frequency(
    angular_frequency: typing.Union[numpy.ndarray, float],
    peak_angular_frequency: float,
    alpha=PHILLIPS_CONSTANT,
    g=GRAVITATIONAL_CONSTANT,
):
    """
    Pierson Moskowitz variance-density spectrum with angular frequency (rad/s) as
    dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

    :param frequency: frequency in rad/s (scalar or array)
    :param peak_frequency: peak frequency in rad/s
    :param alpha: Phillips constant (default 0.0081)
    :param g: gravitational acceleration (default 9.81)
    :return:
    """
    return (
        alpha
        * g**2
        * angular_frequency**-5
        * numpy.exp(-5 / 4 * (peak_angular_frequency / angular_frequency) ** 4)
    )


class FrequencyShape(ABC):
    @abstractmethod
    def values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        pass


class DirectionalShape(ABC):
    @abstractmethod
    def values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        pass


class RaisedCosine(DirectionalShape):
    def __init__(self, mean_direction_degrees: float = 0, width_degrees: float = 28.64):
        self._power = self.power(width_degrees)
        self._mean_direction_degrees = mean_direction_degrees
        self._normalization = (
            numpy.pi
            / 180
            * gamma(self._power / 2 + 1)
            / (gamma(self._power / 2 + 1 / 2) * numpy.sqrt(numpy.pi))
        )

    @staticmethod
    def power(width_degrees):
        return 4 / ((numpy.pi * width_degrees / 90) ** 2) - 2

    def values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        angle = (direction_degrees - self._mean_direction_degrees + 180) % 360 - 180
        with numpy.errstate(invalid="ignore", divide="ignore"):
            return numpy.where(
                numpy.abs(angle) <= 90,
                self._normalization * numpy.cos(angle * numpy.pi / 180) ** self._power,
                0,
            )


class PiersonMoskowitz(FrequencyShape):
    def __init__(
        self, peak_frequency_hertz, m0: float = 1, g: float = GRAVITATIONAL_CONSTANT
    ):
        self._peak_frequency_hertz = peak_frequency_hertz
        self._g = g
        self._alpha = self.alpha(m0)

    def alpha(self, m0):
        return m0 * 5 * (2 * numpy.pi * self._peak_frequency_hertz) ** 4 / self._g**2

    def values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Pierson Moskowitz variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self._peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
        )
        return values


def create_frequency_shape(
    shape: FrequencyShapeOptions, peak_frequency_hertz: float, m0: float = 1
) -> FrequencyShape:
    if shape == "pm":
        return PiersonMoskowitz(peak_frequency_hertz=peak_frequency_hertz, m0=m0)
    else:
        raise ValueError(f"Unknown frequency shape: {shape}")


def create_directional_shape(
    shape: DirectionalShapeOptions,
    mean_direction_degrees: float = 0,
    width_degrees: float = 30,
) -> DirectionalShape:
    if shape == "raised_cosine":
        return RaisedCosine(
            mean_direction_degrees=mean_direction_degrees, width_degrees=width_degrees
        )
    else:
        raise ValueError(f"Unknown frequency shape: {shape}")


def create_parametric_spectrum(
    frequency_hertz: numpy.ndarray,
    frequency_shape: FrequencyShapeOptions,
    peak_frequency_hertz: float,
    significant_wave_height: float,
    direction_degrees: numpy.ndarray,
    direction_shape: DirectionalShapeOptions,
    mean_direction_degrees: float,
    width_degrees: float,
    depth=numpy.inf,
    time: datetime = None,
    latitude: float = None,
    longitude: float = None,
) -> FrequencyDirectionSpectrum:

    m0 = (significant_wave_height / 4) ** 2

    D = create_directional_shape(
        shape=direction_shape,
        mean_direction_degrees=mean_direction_degrees,
        width_degrees=width_degrees,
    ).values(direction_degrees)

    E = create_frequency_shape(
        shape=frequency_shape, peak_frequency_hertz=peak_frequency_hertz, m0=m0
    ).values(frequency_hertz)

    return create_2d_spectrum(
        frequency=frequency_hertz,
        direction=direction_degrees,
        variance_density=E[:, None] * D[None, :],
        time=time,
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        dims=("frequency", "direction"),
    )
