from roguewave import (
    load_spectrum_from_netcdf,
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
)
import importlib.resources


def get_1d_spec() -> FrequencySpectrum:
    file = importlib.resources.open_binary(__package__, "1d_spec.nc")
    return load_spectrum_from_netcdf(file)


def get_2d_spec() -> FrequencyDirectionSpectrum:
    spec1d = get_1d_spec()
    return spec1d.as_frequency_direction_spectrum(36)
