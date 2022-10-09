from typing import Literal
from roguewave import WaveSpectrum
from xarray import DataArray, ones_like
from abc import ABC, abstractmethod
import numpy

_kinematic_viscocity_air = 1.48 * 10**-5

_roughness_length_parameterization = Literal[
    "charnock_constant", "charnock_voermans15", "charnock_voermans16"
]


class RoughnessLength(ABC):
    def z0(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        return self.smooth(friction_velocity) + self.form_drag(
            friction_velocity, spectrum
        )

    @abstractmethod
    def form_drag(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        pass

    def smooth(self, friction_velocity) -> DataArray:
        return 0.11 * _kinematic_viscocity_air / friction_velocity

    @abstractmethod
    def form_drag_derivative_to_ustar(
        self, friction_velocity, spectrum: WaveSpectrum
    ) -> DataArray:
        pass

    def smooth_derivative_to_ustar(self, friction_velocity) -> DataArray:
        return -0.11 * _kinematic_viscocity_air / (friction_velocity**2)

    def derivative_to_ustar(
        self, friction_velocity, spectrum: WaveSpectrum
    ) -> DataArray:
        return self.smooth_derivative_to_ustar(
            friction_velocity
        ) + self.form_drag_derivative_to_ustar(friction_velocity, spectrum)


class CharnockConstant(RoughnessLength):
    def __init__(self, constant=0.012, gravitational_acceleration=9.81):
        self._charnock_constant = constant
        self.gravitational_acceleration = gravitational_acceleration

    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        return self._charnock_constant * ones_like(spectrum.depth)

    def form_drag(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        alpha = self.charnock_constant(friction_velocity, spectrum)
        return alpha * friction_velocity**2 / self.gravitational_acceleration

    def form_drag_derivative_to_ustar(self, friction_velocity, spectrum: WaveSpectrum):
        alpha = self.charnock_constant(friction_velocity, spectrum)
        return 2 * alpha * friction_velocity / self.gravitational_acceleration


class CharnockVoermans15(CharnockConstant):
    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        return 0.06 * (spectrum.hm0() * spectrum.peak_wavenumber) ** 0.7


class CharnockVoermans16(CharnockConstant):
    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        peak_wave_speed = (
            numpy.pi * 2 * spectrum.peak_frequency() / spectrum.peak_wavenumber
        )
        return 0.14 * (friction_velocity / peak_wave_speed) ** 0.61

    def derivative_charnock_constant_to_ustar(
        self, friction_velocity, spectrum: WaveSpectrum
    ) -> DataArray:
        peak_wave_speed = (
            numpy.pi * 2 * spectrum.peak_frequency() / spectrum.peak_wavenumber
        )
        return (
            0.61 * 0.14 * (friction_velocity) ** (-0.39) * (1 / peak_wave_speed) ** 0.61
        )

    def form_drag_derivative_to_ustar(self, friction_velocity, spectrum: WaveSpectrum):
        alpha = self.charnock_constant(friction_velocity, spectrum)
        derivative = self.derivative_charnock_constant_to_ustar(
            friction_velocity, spectrum
        )
        return (
            2 * alpha * friction_velocity / self.gravitational_acceleration
            + friction_velocity**2 / self.gravitational_acceleration * derivative
        )


def create_roughness_length_estimator(
    method: _roughness_length_parameterization = "charnock_constant", **kwargs
) -> RoughnessLength:
    if method == "charnock_constant":
        return CharnockConstant(**kwargs)

    elif method == "charnock_voermans15":
        return CharnockVoermans15(**kwargs)

    elif method == "charnock_voermans16":
        return CharnockVoermans16(**kwargs)
