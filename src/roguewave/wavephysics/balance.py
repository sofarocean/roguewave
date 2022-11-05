from roguewave.wavephysics.generation import (
    WindGeneration,
    wind_parametrizations,
    create_wind_source_term,
)
from .dissipation import (
    Dissipation,
    breaking_parametrization,
    create_breaking_dissipation,
)
from roguewave import FrequencyDirectionSpectrum
from xarray import DataArray


class SourceTermBalance:
    def __init__(self, generation: WindGeneration, disspipation: Dissipation):
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
    generation_par: wind_parametrizations = "st6",
    dissipation_par: breaking_parametrization = "st6",
    generation_args=None,
    dissipation_args=None,
) -> SourceTermBalance:

    dissipation_args = {} if dissipation_args is None else dissipation_args
    generation_args = {} if generation_args is None else generation_args

    return SourceTermBalance(
        generation=create_wind_source_term(generation_par, **generation_args),
        disspipation=create_breaking_dissipation(dissipation_par, **dissipation_args),
    )
