from roguewave.wavespectra.spectrum import (
    FrequencyDirectionSpectrum,
    FrequencySpectrum,
    NAME_D,
    NAME_F,
    SPECTRAL_DIMS,
)
from roguewave.tools.math import wrapped_difference
from typing import Sequence, TypeVar, Union, Literal
from xarray import concat, Dataset, DataArray
from numpy import diff

_T = TypeVar("_T", FrequencySpectrum, FrequencyDirectionSpectrum)
spec_dims = Literal["frequency", "direction"]


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


def integrate_spectral_data(
    dataset: DataArray, dims: Union[spec_dims, Sequence[spec_dims]]
):
    if isinstance(dims, str):
        dims = [dims]

    for dim in dims:
        if dim not in SPECTRAL_DIMS:
            raise ValueError(
                f"Dimension {dim} is not a spectral dimension, options are {NAME_F} or {NAME_D}"
            )

        if dim not in dataset.coords:
            raise ValueError(
                f"Dataset has no {dim} dimension, dimensions are: {list(dataset.dims)}"
            )

    out = dataset
    if NAME_F in dims:
        out = out.fillna(0).integrate(coord=NAME_F)

    if NAME_D in dims:
        difference = DataArray(
            data=wrapped_difference(
                diff(dataset.direction.values, append=dataset.direction[0]), period=360
            ),
            coords={NAME_D: dataset.direction.values},
            dims=[NAME_D],
        )
        out = (out * difference).sum(dim=NAME_D)

    return out
