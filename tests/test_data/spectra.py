from roguewave import load_spectrum_from_netcdf, FrequencySpectrum
import importlib.resources


def get_1d_spec() -> FrequencySpectrum:
    file = importlib.resources.open_binary(__package__, "1d_spec.nc")
    return load_spectrum_from_netcdf(file)


def get_2d_spec() -> FrequencySpectrum:
    file = importlib.resources.open_binary(__package__, "2d_spec.nc")
    return load_spectrum_from_netcdf(file)
