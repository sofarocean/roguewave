from roguewave import FrequencyDirectionSpectrum
from xarray import DataArray
from typing import Literal
from roguewave.wavephysics.dissipation.base_class import WaveBreaking
from numpy.typing import NDArray

breaking_parametrization = Literal["st6", "st4"]


class ST4(WaveBreaking):
    def __init__(self):
        pass

    def rate(self, spectrum: FrequencyDirectionSpectrum, memoize=None) -> DataArray:
        k = spectrum.wavenumber
        e = spectrum.e
        cg = spectrum.group_velocity
        relative_exceedence_level = self.relative_exceedence_level(e, k, cg)

        inherent = self.inherent(spectrum, relative_exceedence_level)
        induced = self.induced(spectrum, relative_exceedence_level)
        return inherent + induced

    def inherent(self, spectrum: FrequencyDirectionSpectrum, relative_exceedence_level):
        pass

    def induced(self, spectrum: FrequencyDirectionSpectrum, relative_exceedence_level):
        pass

    def spectral_threshold(self, wavenumber):
        pass

    def relative_exceedence_level(self, e, wavenumber, cg):
        pass


def partially_integrated_saturation(
    spectrum: NDArray, theta, wavenumber, directions, direction_increment
):
    pass
