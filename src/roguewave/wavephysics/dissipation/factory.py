from roguewave.wavephysics.dissipation.base_class import Dissipation
from typing import Literal
from roguewave.wavephysics.dissipation.st6_wave_breaking import ST6WaveBreaking
from roguewave.wavephysics.dissipation.st4_wave_breaking import ST4WaveBreaking


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
