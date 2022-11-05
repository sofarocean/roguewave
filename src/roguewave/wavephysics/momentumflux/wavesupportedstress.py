from roguewave.wavephysics.generation import WindGeneration, TWindInputType
from roguewave.wavespectra import FrequencyDirectionSpectrum
from xarray import DataArray
from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from numpy import cos, pi, sqrt, sin, empty, arctan2
from numpy.typing import NDArray
from numba import njit
from typing import Tuple


def wave_supported_stress(
    spectrum: FrequencyDirectionSpectrum,
    speed: DataArray,
    wind_direction,
    roughness_length: DataArray,
    generation: WindGeneration,
    wind_input_type: TWindInputType = "u10",
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
    memoized=None,
) -> Tuple[DataArray, DataArray]:
    """

    :param spectrum:
    :param u10:
    :param wind_direction:
    :param air:
    :param water:
    :param gravitational_acceleration:
    :return:
    """
    if memoized is None:
        memoized = {}

    if "wave_speed" not in memoized:
        memoized["wave_speed"] = spectrum.wave_speed()

    if "wind_mutual_angle" not in memoized:
        memoized["wind_mutual_angle"] = (
            ((spectrum.direction - wind_direction + 180.0) % 360.0 - 180.0) * pi / 180
        )

    if "wavenumber" not in memoized:
        memoized["wavenumber"] = spectrum.wavenumber

    wind_input = generation.rate(
        speed, wind_direction, spectrum, roughness_length, wind_input_type
    )

    stress_magnitude, stress_direction = numba_wave_supported_stress(
        wind_input.values,
        spectrum.radian_direction.values,
        spectrum.radian_frequency.values,
        memoized["wavenumber"].values,
        spectrum.number_of_directions,
        spectrum.number_of_frequencies,
        spectrum.number_of_spectra,
        spectrum.frequency_step.values,
        spectrum.direction_step.values,
        water.density,
        gravitational_acceleration,
    )

    coords = {dim: spectrum.dataset[dim] for dim in spectrum.dims_space_time}
    magnitude = DataArray(
        data=stress_magnitude, dims=spectrum.dims_space_time, coords=coords
    )
    direction = DataArray(
        data=stress_direction, dims=spectrum.dims_space_time, coords=coords
    )
    return magnitude, direction


@njit(cache=True)
def numba_wave_supported_stress(
    wind_input: NDArray,
    radian_direction: NDArray,
    radian_frequency: NDArray,
    wavenumber: NDArray,
    number_of_directions,
    number_of_frequencies,
    number_of_points,
    frequency_step,
    direction_step,
    water_density: float,
    gravitational_acceleration,
) -> Tuple[NDArray, NDArray]:
    """
    :param spectrum:
    :param radian_direction:
    :param radian_frequency:
    :param depth:
    :param wind_input:
    :param water_density:
    :param number_of_directions:
    :param number_of_frequencies:
    :param number_of_points:
    :param frequency_step:
    :param direction_step:
    :param gravitational_acceleration:
    :param memoized:
    :return:
    """

    wave_stress_magnitude = empty((number_of_points,))
    wave_stress_direction = empty((number_of_points,))
    common_factor = gravitational_acceleration * water_density
    for point_index in range(number_of_points):
        stress_east = 0.0
        stress_north = 0.0

        for frequency_index in range(number_of_frequencies):
            inverse_wave_speed = (
                wavenumber[point_index, frequency_index]
                / radian_frequency[frequency_index]
            )
            for direction_index in range(number_of_directions):
                stress_east += (
                    cos(radian_direction[direction_index])
                    * wind_input[point_index, frequency_index, direction_index]
                    * inverse_wave_speed
                    * frequency_step[frequency_index]
                    * direction_step[direction_index]
                )

                stress_north += (
                    sin(radian_direction[direction_index])
                    * wind_input[point_index, frequency_index, direction_index]
                    * inverse_wave_speed
                    * frequency_step[frequency_index]
                    * direction_step[direction_index]
                )
        wave_stress_magnitude[point_index] = (
            sqrt(stress_north**2 + stress_east**2) * common_factor
        )
        wave_stress_direction[point_index] = (
            arctan2(stress_north, stress_east) * 180 / pi
        ) % 360

    return wave_stress_magnitude, wave_stress_direction
