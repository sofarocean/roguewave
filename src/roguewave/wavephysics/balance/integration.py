from numba import njit
from numpy.typing import NDArray
from numpy import empty
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