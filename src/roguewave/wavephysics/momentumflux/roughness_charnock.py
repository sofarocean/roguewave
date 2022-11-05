from roguewave import WaveSpectrum
from xarray import DataArray, ones_like
from roguewave.wavephysics.fluidproperties import GRAVITATIONAL_ACCELERATION
from roguewave.wavephysics.momentumflux.roughness_base_class import RoughnessLength
import numpy


class CharnockConstant(RoughnessLength):
    def __init__(
        self,
        charnock_constant=0.012,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
    ):
        self._charnock_constant = charnock_constant
        self.gravitational_acceleration = gravitational_acceleration
        self.charnock_maximum = numpy.inf

    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        return self._charnock_constant * ones_like(spectrum.depth)

    def form_drag(
        self, friction_velocity, spectrum: WaveSpectrum, direction_degrees
    ) -> DataArray:
        alpha = self.charnock_constant(friction_velocity, spectrum)
        return numpy.minimum(
            alpha * friction_velocity**2 / self.gravitational_acceleration,
            self.charnock_maximum,
        )


class CharnockVoermans15(CharnockConstant):
    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        return 0.06 * (spectrum.hm0() * spectrum.peak_wavenumber) ** 0.7


class CharnockVoermans16(CharnockConstant):
    def charnock_constant(self, friction_velocity, spectrum: WaveSpectrum) -> DataArray:
        peak_wave_speed = (
            numpy.pi * 2 * spectrum.peak_frequency() / spectrum.peak_wavenumber
        )
        return 0.14 * (friction_velocity / peak_wave_speed) ** 0.61
