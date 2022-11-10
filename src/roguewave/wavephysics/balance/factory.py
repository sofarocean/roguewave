from roguewave.wavephysics.balance.dissipation import Dissipation
from typing import Literal
from roguewave.wavephysics.balance.st6_wave_breaking import ST6WaveBreaking
from roguewave.wavephysics.balance.st4_wave_breaking import ST4WaveBreaking

from roguewave.wavephysics.balance.st4_wind_input import ST4WindInput
from roguewave.wavephysics.balance.st6_wind_input import ST6WindInput

from roguewave.wavephysics.balance.balance import SourceTermBalance

breaking_parametrization = Literal["st6", "st4"]


def create_breaking_dissipation(
    breaking_parametrization: breaking_parametrization = "st6", **kwargs
) -> Dissipation:
    if breaking_parametrization == "st6":
        return ST6WaveBreaking(**kwargs)
    elif breaking_parametrization == "st4":
        return ST4WaveBreaking(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization {breaking_parametrization}")


wind_parametrizations = Literal["st6", "st4", "st4_swell"]


def create_wind_source_term(
    wind_parametrization: wind_parametrizations = "st6", **kwargs
):
    if wind_parametrization == "st6":
        return ST6WindInput(**kwargs)
    elif wind_parametrization == "st4":
        return ST4WindInput(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization: {wind_parametrization}")


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
