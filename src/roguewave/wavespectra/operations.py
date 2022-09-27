from roguewave.wavespectra.spectrum import FrequencyDirectionSpectrum, FrequencySpectrum
from typing import Sequence, TypeVar
from xarray import concat, Dataset

_T = TypeVar("_T", FrequencySpectrum, FrequencyDirectionSpectrum)


def concatenate_spectra(spectra: Sequence[_T], dim, **kwargs) -> _T:
    """
    Concatenate along the given dimension. If the dimension does not exist a new dimension will be created. Under the
    hood this calls the concat function of xarray. Named arguments to that function can be applied here as well.

    :param spectra: A sequence of Frequency Spectra/Frequency Direction Spectra
    :param dim: the dimension to concatenate along
    :return: New combined spectral object.
    """

    # Concatenate the dataset in the spectral objects using the xarray concatenate function
    dataset = Dataset()
    dataset["variance_density"] = concat(
        [x.dataset["variance_density"] for x in spectra], dim=dim, **kwargs
    )
    dataset["depth"] = spectra[0].depth
    dataset["latitude"] = spectra[0].latitude
    dataset["longitude"] = spectra[0].longitude
    if "time" not in dataset:
        dataset["time"] = spectra[0].time

    # Get the class of the input spectra.
    cls = type(spectra[0])

    # Return a class instance.
    return cls(dataset)
