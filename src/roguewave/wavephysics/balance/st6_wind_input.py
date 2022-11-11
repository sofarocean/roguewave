from roguewave.wavephysics.fluidproperties import AIR, WATER, GRAVITATIONAL_ACCELERATION
from roguewave.wavespectra.operations import numba_directionally_integrate_spectral_data
from roguewave.wavetheory import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_velocity,
)
from roguewave.wavephysics.balance import WindGeneration
from numpy import tanh, cos, pi, sqrt, log, empty, argmax, inf
from numba import njit
from typing import TypedDict


class ST6WaveGenerationParameters(TypedDict):
    gravitational_acceleration: float
    charnock_maximum_roughness: float
    charnock_constant: float
    air_density: float
    water_density: float
    vonkarman_constant: float
    friction_velocity_scaling: float
    elevation: float


class ST6WindInput(WindGeneration):
    name = "st6 generation"

    def __init__(
        self,
        parameters: ST6WaveGenerationParameters = None,
    ):
        super(ST6WindInput, self).__init__(parameters)
        self._wind_source_term_function = _st6_wind_generation_point

    @staticmethod
    def default_parameters() -> ST6WaveGenerationParameters:
        return ST6WaveGenerationParameters(
            gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
            charnock_maximum_roughness=inf,
            charnock_constant=0.015,
            air_density=AIR.density,
            water_density=WATER.density,
            vonkarman_constant=AIR.vonkarman_constant,
            friction_velocity_scaling=28,
            elevation=10,
        )


# ----------------------------------------------------------------------------------------------------------------------
# st6 Point wise implementation functions (apply to a single spatial point)
# ----------------------------------------------------------------------------------------------------------------------


@njit(cache=True)
def _st6_wind_generation_point(
    variance_density,
    wind,
    depth,
    roughness_length,
    st6_spectral_grid,
    st6_parameters,
    wind_source=None,
):
    """
    Implememtation of the st6 wind growth rate term

    :param variance_density:
    :param wind:
    :param depth:
    :param roughness_length:
    :param st6_spectral_grid:
    :param st6_parameters:
    :return:
    """

    # Get the spectral size
    number_of_frequencies, number_of_directions = variance_density.shape

    # Allocate the output variable
    if wind_source is None:
        wind_source = empty((number_of_frequencies, number_of_directions))

    # Unpack variables passed in dictionaries/tuples for ease of reference and some slight speed benifits (avoid dictionary
    # lookup in loops).
    vonkarman_constant = st6_parameters["vonkarman_constant"]
    friction_velocity_scaling = st6_parameters["friction_velocity_scaling"]
    air_density = st6_parameters["air_density"]
    water_density = st6_parameters["water_density"]
    elevation = st6_parameters["elevation"]
    radian_frequency = st6_spectral_grid["radian_frequency"]
    radian_direction = st6_spectral_grid["radian_direction"]
    wind_forcing, wind_direction_degrees, wind_forcing_type = wind

    # Preprocess winds to the correct conventions (U10->friction velocity if need be, wind direction degrees->radians)
    if wind_forcing_type == "u10":
        u10 = wind_forcing
        friction_velocity = u10 * vonkarman_constant / log(elevation / roughness_length)

    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        u10 = wind_forcing / vonkarman_constant * log(elevation / roughness_length)
        friction_velocity = wind_forcing

    else:
        raise ValueError("unknown wind forcing")
    wind_direction_radian = wind_direction_degrees * pi / 180
    us = friction_velocity * friction_velocity_scaling
    frequency_spectrum = numba_directionally_integrate_spectral_data(
        variance_density, st6_spectral_grid
    )
    wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    wavespeed = radian_frequency / wavenumber
    group_velocity = intrinsic_group_velocity(wavenumber, depth)
    saturation_spectrum = frequency_spectrum * group_velocity * wavenumber**3 / 2 / pi

    peak_index = argmax(frequency_spectrum)
    peak_wave_speed = wavespeed[peak_index]
    peak_radian_frequency = radian_frequency[peak_index]

    # precalculate the constant multiplication.
    constant_factor = air_density / water_density

    # Loop over all frequencies/directions
    for frequency_index in range(number_of_frequencies):

        # Since wavenumber and wavespeed only depend on frequency we calculate those outside of the direction loop.
        constant_frequency_factor = constant_factor * radian_frequency[frequency_index]

        st6_directional_spreading_function = 1.12 * (us / peak_wave_speed) ** (-0.5) * (
            radian_frequency[frequency_index] / peak_radian_frequency
        ) ** -(0.95) + 1 / (2 * pi)

        root_of_saturation = sqrt(
            saturation_spectrum[frequency_index] * st6_directional_spreading_function
        )

        for direction_index in range(number_of_directions):
            # Calculate the directional factor in the wind input formulation
            W = (
                u10
                * cos(radian_direction[direction_index] - wind_direction_radian)
                / wavespeed[frequency_index]
                - 1
            )

            if W > 0:
                # If the wave direction has an along wind component we continue.

                directional_saturation = W**2 * root_of_saturation

                st6_sheltering_coefficient = 2.8 - (
                    1 + tanh(10 * directional_saturation - 11)
                )
                growth_rate = (
                    constant_frequency_factor
                    * st6_sheltering_coefficient
                    * directional_saturation
                )

            else:
                growth_rate = 0.0

            wind_source[frequency_index, direction_index] = (
                growth_rate * variance_density[frequency_index, direction_index]
            )
    return wind_source
