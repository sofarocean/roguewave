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
    dataset[dim] = [x.dataset[dim].values for x in spectra]

    for variable_name in spectra[0]:
        if variable_name == dim:
            continue

        dataset[variable_name] = concat(
            [x.dataset[variable_name] for x in spectra], dim=dim, **kwargs
        )

    # Get the class of the input spectra.
    cls = type(spectra[0])

    # Return a class instance.
    return cls(dataset)
