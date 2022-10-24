from roguewave.wavephysics.generation import (
    WindGeneration,
    wind_parametrizations,
    create_wind_source_term,
)
from .dissipation import (
    WaveBreaking,
    breaking_parametrization,
    create_breaking_dissipation,
)
from roguewave import FrequencyDirectionSpectrum
from xarray import DataArray


class SourceTermBalance:
    def __init__(self, generation: WindGeneration, disspipation: WaveBreaking):
        self.generation = generation
        self.dissipation = disspipation

    def evaluate_imbalance(
        self,
        wind_speed: DataArray,
        wind_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        return self.generation.rate(
            wind_speed, wind_direction, spectrum
        ) + self.dissipation.rate(spectrum)

    def evaluate_bulk_imbalance(
        self,
        wind_speed: DataArray,
        wind_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        return self.generation.bulk_rate(
            wind_speed, wind_direction, spectrum
        ) + self.dissipation.bulk_rate(spectrum)
        #
        # return integrate_spectral_data(
        #     dataset=self.evaluate_imbalance(wind_speed,wind_direction,spectrum),
        #     dims=["frequency","direction"]
        # )


def create_balance(
    wind_par: wind_parametrizations = "st6",
    wave_par: breaking_parametrization = "st6",
    wind_args=None,
    wave_args=None,
) -> SourceTermBalance:

    wave_args = {} if wave_args is None else wave_args
    wind_args = {} if wind_args is None else wind_args

    return SourceTermBalance(
        generation=create_wind_source_term(wind_par, **wind_args),
        disspipation=create_breaking_dissipation(wave_par, **wave_args),
    )
