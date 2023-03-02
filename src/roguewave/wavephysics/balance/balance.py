from roguewave.wavephysics.balance import WindGeneration, Dissipation
from typing import Mapping, Dict
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
        time_derivative_spectrum: FrequencyDirectionSpectrum = None,
    ) -> DataArray:

        if time_derivative_spectrum is None:
            time_derivative_spectrum = 0.0
        else:
            time_derivative_spectrum = time_derivative_spectrum.variance_density

        return (
            self.generation.rate(
                spectrum,
                wind_speed,
                wind_direction,
            )
            + self.dissipation.rate(spectrum)
            - time_derivative_spectrum
        )

    def evaluate_bulk_imbalance(
        self,
        wind_speed: DataArray,
        wind_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        time_derivative_spectrum: FrequencyDirectionSpectrum = None,
    ) -> DataArray:

        if time_derivative_spectrum is None:
            time_derivative_spectrum = 0.0
        else:
            time_derivative_spectrum = time_derivative_spectrum.m0()

        return (
            self.generation.bulk_rate(spectrum, wind_speed, wind_direction)
            + self.dissipation.bulk_rate(spectrum)
            - time_derivative_spectrum
        )

    def update_parameters(self, parameters: Mapping):
        self.generation.update_parameters(parameters)
        self.dissipation.update_parameters(parameters)

    @property
    def get_parameters(self) -> Dict:
        return self.generation._parameters | self.dissipation._parameters
