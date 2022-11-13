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
from numpy import diff, empty
from numpy.typing import NDArray
from numba import njit

_T = TypeVar("_T", FrequencySpectrum, FrequencyDirectionSpectrum)
spec_dims = Literal["frequency", "direction"]


def concatenate_spectra(spectra: Sequence[_T], dim=None, keys=None, **kwargs) -> _T:
    """
    Concatenate along the given dimension. If the dimension does not exist a new dimension will be created. Under the
    hood this calls the concat function of xarray. Named arguments to that function can be applied here as well.

    If dim is set to None - we first flatten the spectral objects - and then join along the flattened dimension.

    :param spectra: A sequence of Frequency Spectra/Frequency Direction Spectra
    :param dim: the dimension to concatenate along
    :return: New combined spectral object.
    """

    # Concatenate the dataset in the spectral objects using the xarray concatenate function
    dataset = Dataset()

    if dim is not None:
        # Why are we doing this?
        dataset[dim] = [x.dataset[dim].values for x in spectra]
    else:
        dim = "linear_index"
        spectra = [x.flatten(dim) for x in spectra]

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


@njit(cache=True, fastmath=True)
def numba_integrate_spectral_data(data: NDArray, grid):

    frequency_step = grid["frequency_step"]
    direction_step = grid["direction_step"]
    number_of_frequencies = data.shape[-2]
    number_of_directions = data.shape[-1]

    return_value = 0.0
    for frequency_index in range(number_of_frequencies):
        for direction_index in range(number_of_directions):
            return_value += (
                data[frequency_index, direction_index]
                * frequency_step[frequency_index]
                * direction_step[direction_index]
            )
    return return_value


@njit(cache=True, fastmath=True)
def numba_directionally_integrate_spectral_data(data: NDArray, grid):

    direction_step = grid["direction_step"]
    number_of_frequencies = data.shape[-2]
    number_of_directions = data.shape[-1]

    return_value = empty(number_of_frequencies, dtype=data.dtype)
    for frequency_index in range(number_of_frequencies):
        return_value[frequency_index] = 0.0
        for direction_index in range(number_of_directions):
            return_value[frequency_index] += (
                data[frequency_index, direction_index] * direction_step[direction_index]
            )
    return return_value
