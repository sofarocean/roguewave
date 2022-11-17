from typing import TypedDict


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
            sampling_interval_gps=0.4,
            sampling_interval_spectra=3600,
            sampling_interval_location=3600,
            number_of_samples=256,
            number_of_frequencies=128,
            n_channel=False,
        )
    else:
        raise ValueError("unknown settings")
