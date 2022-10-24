from typing import Literal
from roguewave.wavephysics.generation.st4 import ST4
from roguewave.wavephysics.generation.st6 import ST6

wind_parametrizations = Literal["st6", "st4"]


def create_wind_source_term(
    wind_parametrization: wind_parametrizations = "st6", **kwargs
):
    if wind_parametrization == "st6":
        return ST6(**kwargs)
    elif wind_parametrization == "st4":
        return ST4(**kwargs)
    else:
        raise ValueError(f"Unknown parametrization: {wind_parametrization}")
