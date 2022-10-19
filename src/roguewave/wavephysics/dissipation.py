from roguewave import FrequencyDirectionSpectrum, integrate_spectral_data
from abc import ABC, abstractmethod
from xarray import DataArray, zeros_like, where
from numpy import diff, pi
from typing import Literal

breaking_parametrization = Literal["st6", "st4"]


class WaveBreaking(ABC):
    @abstractmethod
    def rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:
        pass

    def saturation_spectrum(self, spectrum, cg, k):
        return cg * spectrum.e * k**3 / 2 / pi

    def bulk_rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:
        frequency: Literal[
            "frequency"
        ] = "frequency"  # To avoid issues with pycharm flagging the argument as not being a literal
        direction: Literal["direction"] = "direction"
        return integrate_spectral_data(self.rate(spectrum), dims=[frequency, direction])


class ST6(WaveBreaking):
    def __init__(
        self,
        p1=4,
        p2=4,
        a1=4.75 * 10**-6,
        a2=7e-5,
        dimensionless_saturation_threshold=0.035**2,
    ):
        self.dimensionless_saturation_threshold = dimensionless_saturation_threshold
        self.p1 = p1
        self.p2 = p2
        self.a1 = a1
        self.a2 = a2

    def rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:
        k = spectrum.wavenumber
        cg = spectrum.group_velocity
        saturation_spectrum = self.saturation_spectrum(spectrum, cg, k)
        relative_exceedence_level = self.relative_exceedence_level(saturation_spectrum)

        inherent = self.inherent(spectrum, relative_exceedence_level)
        induced = self.induced(spectrum, relative_exceedence_level)
        return inherent + induced

    def inherent(self, spectrum: FrequencyDirectionSpectrum, relative_exceedence_level):
        return (
            -self.a1
            * relative_exceedence_level**self.p1
            * spectrum.frequency
            * spectrum.variance_density
        )

    def induced(self, spectrum: FrequencyDirectionSpectrum, relative_exceedence_level):
        delta = diff(spectrum.frequency)
        freq_delta = zeros_like(spectrum.frequency)
        freq_delta[0:-1] += delta * 0.5
        freq_delta[1:] += delta * 0.5
        freq_delta[0] += delta[0] * 0.5
        freq_delta[-1] += delta[-1] * 0.5
        return (
            -self.a2
            * (
                (freq_delta * relative_exceedence_level.fillna(0)).cumsum(
                    dim="frequency"
                )
            )
            ** self.p2
            * spectrum.variance_density
        )

    def relative_exceedence_level(self, saturation_spectrum):
        delta = (
            saturation_spectrum - self.dimensionless_saturation_threshold
        ) / self.dimensionless_saturation_threshold
        return where(delta > 0, delta, 0)


class ST4(WaveBreaking):
    def __init__(self):
        pass

    def rate(self, spectrum: FrequencyDirectionSpectrum) -> DataArray:
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


def create_breaking_dissipation(
    breaking_parametrization: breaking_parametrization = "st6", **kwargs
) -> WaveBreaking:
    if breaking_parametrization == "st6":
        return ST6(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization {breaking_parametrization}")
