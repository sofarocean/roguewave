from roguewave.wavephysics.dissipation.base_class import WaveBreaking
from typing import Literal
from roguewave.wavephysics.dissipation.st6 import ST6
from roguewave.wavephysics.dissipation.st4 import ST4


breaking_parametrization = Literal["st6", "st4"]


def create_breaking_dissipation(
    breaking_parametrization: breaking_parametrization = "st6", **kwargs
) -> WaveBreaking:
    if breaking_parametrization == "st6":
        return ST6(**kwargs)
    elif breaking_parametrization == "st4":
        return ST4(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization {breaking_parametrization}")
