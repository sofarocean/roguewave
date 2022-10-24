from typing import Literal
from roguewave.wavephysics.momentumflux.roughness_base_class import RoughnessLength
from roguewave.wavephysics.momentumflux.roughness_charnock import (
    CharnockConstant,
    CharnockVoermans15,
    CharnockVoermans16,
)
from roguewave.wavephysics.momentumflux.roughness_janssen import Janssen

_roughness_length_parameterization = Literal[
    "charnock_constant",
    "charnock_voermans15",
    "charnock_voermans16",
    "charnock_janssen",
]


def create_roughness_length_estimator(
    method: _roughness_length_parameterization = "charnock_constant", **kwargs
) -> RoughnessLength:
    if method == "charnock_constant":
        return CharnockConstant(**kwargs)

    elif method == "charnock_voermans15":
        return CharnockVoermans15(**kwargs)

    elif method == "charnock_voermans16":
        return CharnockVoermans16(**kwargs)

    elif method == "charnock_janssen":
        return Janssen(**kwargs)
