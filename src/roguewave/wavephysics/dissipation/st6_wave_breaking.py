from roguewave import FrequencyDirectionSpectrum
from xarray import DataArray, zeros_like, where
from numpy import diff, pi
from typing import Literal
from roguewave.wavephysics.dissipation.base_class import Dissipation

breaking_parametrization = Literal["st6", "st4"]


class ST6WaveBreaking(Dissipation):
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

    def saturation_spectrum(self, spectrum, cg, k, memoize=None):
        if memoize is None:
            memoize = {}

        if "saturation_spectrum" not in memoize:
            memoize["saturation_spectrum"] = cg * spectrum.e * k**3 / 2 / pi
        return memoize["saturation_spectrum"]

    def rate(self, spectrum: FrequencyDirectionSpectrum, memoize=None) -> DataArray:

        if memoize is None:
            memoize = {}

        if "wavenumber" not in memoize:
            memoize["wavenumber"] = spectrum.wavenumber

        if "group_velocity" not in memoize:
            memoize["group_velocity"] = spectrum.group_velocity

        saturation_spectrum = self.saturation_spectrum(
            spectrum, memoize["group_velocity"], memoize["wavenumber"]
        )
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
