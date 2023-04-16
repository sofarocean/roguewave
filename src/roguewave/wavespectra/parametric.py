from .spectrum import FrequencyDirectionSpectrum, create_2d_spectrum, FrequencySpectrum
from abc import ABC, abstractmethod
from scipy.special import gamma
from datetime import datetime
import numpy
from typing import Literal

PHILLIPS_CONSTANT = 0.0081
GRAVITATIONAL_CONSTANT = 9.81

FrequencyShapeOptions = Literal["pm", "jonswap", "phillips", "gaussian"]
DirectionalShapeOptions = Literal["raised_cosine"]


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


class GaussianSpectrum(FrequencyShape):
    def __init__(self, peak_frequency_hertz, m0: float = 1, **kwargs):
        self.m0 = m0
        self._peak_frequency_hertz = peak_frequency_hertz
        self.standard_deviation_hertz = kwargs.get(
            "standard_deviation_hertz", peak_frequency_hertz / 10
        )

    def values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        return (
            self.m0
            / self.standard_deviation_hertz
            / numpy.sqrt(2 * numpy.pi)
            * numpy.exp(
                -0.5
                * (frequency_hertz - self._peak_frequency_hertz) ** 2
                / self.standard_deviation_hertz**2
            )
        )


class PhillipsSpectrum(FrequencyShape):
    def __init__(self, peak_frequency_hertz, m0: float = 1, **kwargs):

        self._peak_frequency_hertz = peak_frequency_hertz
        self._g = kwargs.get("g", GRAVITATIONAL_CONSTANT)
        self._alpha = self.alpha(m0)

    def alpha(self, m0):
        return m0 * 8 * (numpy.pi) ** 4 * self._peak_frequency_hertz**4 / self._g**2

    def values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Phillips variance-density spectrum with frequency in Hz as
        dependent variable.

        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
        )
        return values


class PiersonMoskowitzSpectrum(FrequencyShape):
    def __init__(self, peak_frequency_hertz, m0: float = 1, **kwargs):
        self._peak_frequency_hertz = peak_frequency_hertz
        self._g = kwargs.get("g", GRAVITATIONAL_CONSTANT)
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


class JonswapSpectrum(FrequencyShape):
    def __init__(self, peak_frequency_hertz, m0: float = 1, **kwargs):
        self._peak_frequency_hertz = peak_frequency_hertz
        self._g = kwargs.get("g", GRAVITATIONAL_CONSTANT)
        self._sigma_a = kwargs.get("sigma_a", 0.07)
        self._sigma_b = kwargs.get("sigma_b", 0.09)
        self.gamma = kwargs.get("gamma", 3.3)
        self._alpha = self.alpha(m0)

    def alpha(self, m0):
        # Approximation by Yamaguchi (1984), "Approximate expressions for integral properties of the JONSWAP
        # spectrum" Proc. Japanese Society of Civil Engineers, 345/II-1, 149–152 [in Japanese]. Taken from Holthuijsen
        # "waves in oceanic and coastal waters". Not valid if sigma_a or sigma_b are chanegd from defaults. Otherwise
        # accurate to within 0.25%
        #
        return (
            m0
            * (2 * numpy.pi * self._peak_frequency_hertz) ** 4
            / self._g**2
            / (0.06533 * self.gamma**0.8015 + 0.13467)
        )

    def values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Jonswap variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0

        sigma = numpy.where(
            frequency_hertz <= self._peak_frequency_hertz, self._sigma_a, self._sigma_b
        )
        peak_enhancement = self.gamma ** numpy.exp(
            -1 / 2 * ((frequency_hertz / self._peak_frequency_hertz - 1) / sigma) ** 2
        )

        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self._peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
            * peak_enhancement[msk]
        )
        return values


def create_frequency_shape(
    shape: FrequencyShapeOptions, peak_frequency_hertz: float, m0: float = 1, **kwargs
) -> FrequencyShape:
    if shape == "pm":
        return PiersonMoskowitzSpectrum(
            peak_frequency_hertz=peak_frequency_hertz, m0=m0, **kwargs
        )
    elif shape == "jonswap":
        return JonswapSpectrum(
            peak_frequency_hertz=peak_frequency_hertz, m0=m0, **kwargs
        )
    elif shape == "phillips":
        return PhillipsSpectrum(
            peak_frequency_hertz=peak_frequency_hertz, m0=m0, **kwargs
        )
    elif shape == "gaussian":
        return GaussianSpectrum(
            peak_frequency_hertz=peak_frequency_hertz, m0=m0, **kwargs
        )

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


def create_parametric_frequency_direction_spectrum(
    frequency_hertz: numpy.ndarray,
    peak_frequency_hertz: float,
    significant_wave_height: float,
    frequency_shape: FrequencyShapeOptions = "jonswap",
    direction_degrees: numpy.ndarray = None,
    direction_shape: DirectionalShapeOptions = "raised_cosine",
    mean_direction_degrees: float = 0.0,
    width_degrees: float = 30,
    depth=numpy.inf,
    time: datetime = None,
    latitude: float = None,
    longitude: float = None,
    **kwargs,
) -> FrequencyDirectionSpectrum:
    """
    Create a parametrized directional frequency spectrum according to a given frequency (Jonswap, PM) or directional
    (raised_cosine) distribution.

    :param frequency_hertz: Frequencies to resolve
    :param peak_frequency_hertz:  Desired peak frequency of the spectrum
    :param significant_wave_height: Significant wave height of the spectrum
    :param frequency_shape: The frequency shape, currently supported are:
        frequency_shape="pm": for pierson_moskowitz
        frequency_shape="jonswap" [default]: for Jonswap
    :param direction_degrees: Directions to resolve the spectrum. If None [default] 36 directions spanning the circle
        are used [ 0 , 360 )
    :param direction_shape: shape of the directional distribution. Currently only a raised cosine distribution is
        supported.
    :param mean_direction_degrees: mean direction of the waves. 0 degrees (due east) is the default.
    :param width_degrees: width of the spectrum (according to Kuik). 30 degrees is the default.
    :param depth: mean depth at the location of the spectrum (optional). Does not affect returned spectral values in any
        way, but is used as the depth in the returned spectral object (and may affect e.g. wavenumber calculations.)
    :param time: timestamp of the spectrum. Optional. Merely an annotation on the returned object.
    :param latitude: latitude of the spectrum. Optional. Merely an annotation on the returned object.
    :param longitude: latitude of the spectrum. Optional. Merely an annotation on the returned object.

    :return: FrequencyDirectionSpectrum object.
    """

    if direction_degrees is None:
        direction_degrees = numpy.linspace(0, 360, 36, endpoint=False)

    D = create_directional_shape(
        shape=direction_shape,
        mean_direction_degrees=mean_direction_degrees,
        width_degrees=width_degrees,
    ).values(direction_degrees)

    m0 = (significant_wave_height / 4) ** 2
    E = create_frequency_shape(
        shape=frequency_shape,
        peak_frequency_hertz=peak_frequency_hertz,
        m0=m0,
        **kwargs,
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


def create_parametric_frequency_spectrum(
    frequency_hertz: numpy.ndarray,
    peak_frequency_hertz: float,
    significant_wave_height: float,
    frequency_shape: FrequencyShapeOptions = "jonswap",
    depth=numpy.inf,
    time: datetime = None,
    latitude: float = None,
    longitude: float = None,
    **kwargs,
) -> FrequencySpectrum:

    # We create a 1d spectrum from an integrated 2d spectrum with assumed raised cosine shape. This allows us to
    # add the a1/b1 parameters easily.
    spec2d = create_parametric_frequency_direction_spectrum(
        frequency_hertz,
        peak_frequency_hertz,
        significant_wave_height,
        frequency_shape,
        depth=depth,
        time=time,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )
    return spec2d.as_frequency_spectrum()


def create_parametric_spectrum(
    frequency_hertz: numpy.ndarray,
    frequency_shape: FrequencyShapeOptions,
    peak_frequency_hertz: float,
    significant_wave_height: float,
    direction_degrees: numpy.ndarray = None,
    direction_shape: DirectionalShapeOptions = "raised_cosine",
    mean_direction_degrees: float = 0.0,
    width_degrees: float = 30.0,
    depth=numpy.inf,
    time: datetime = None,
    latitude: float = None,
    longitude: float = None,
) -> FrequencyDirectionSpectrum:
    """
    Deprecated - use create_parametric_frequency_direction_spectrum instead
    """
    return create_parametric_frequency_direction_spectrum(
        frequency_hertz,
        peak_frequency_hertz,
        significant_wave_height,
        frequency_shape,
        direction_degrees,
        direction_shape,
        mean_direction_degrees,
        width_degrees,
        depth,
        time,
        latitude,
        longitude,
    )
