from typing import TypedDict
from numpy import timedelta64


class SpotterConstants(TypedDict):
    sampling_interval_gps: float
    sampling_interval_spectra: float
    sampling_interval_location: float
    number_of_samples: int
    number_of_frequencies: int
    n_channel: bool


def spotter_constants(settings="default") -> SpotterConstants:
    if settings == "default":
        return SpotterConstants(
            sampling_interval_gps=timedelta64(400, "ms"),
            sampling_interval_spectra=timedelta64(3600, "s"),
            sampling_interval_location=timedelta64(60, "s"),
            number_of_samples=256,
            number_of_frequencies=128,
            n_channel=False,
        )
    else:
        raise ValueError("unknown settings")
