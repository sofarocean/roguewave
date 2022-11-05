from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from roguewave import FrequencyDirectionSpectrum
from roguewave.wavephysics.generation import WindGeneration, TWindInputType
from numpy import cos, pi, log, exp, empty, arctan2, sin, sqrt, inf, isnan, nan, any
from xarray import DataArray, zeros_like
from numba import njit
from numpy.typing import NDArray
from roguewave.wavetheory import inverse_intrinsic_dispersion_relation
from roguewave.wavespectra.operations import numba_integrate_spectral_data
from typing import Tuple
from numba_progress import ProgressBar
from roguewave.tools.solvers import numba_fixed_point_iteration, numba_newton_raphson


class ST4(WindGeneration):
    def __init__(
        self,
        wave_age_tuning_parameter=0.006,
        growth_parameter_betamax=1.8,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
        charnock_maximum_roughness=inf,
        charnock_constant=0.015,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        **kwargs
    ):
        super(ST4, self).__init__(**kwargs)

        self.parameters = _st4_parameters(
            air_density=air.density,
            water_density=water.density,
            wave_age_tuning_parameter=wave_age_tuning_parameter,
            growth_parameter_betamax=growth_parameter_betamax,
            gravitational_acceleration=gravitational_acceleration,
            charnock_maximum_roughness=charnock_maximum_roughness,
            charnock_constant=charnock_constant,
            vonkarman_constant=AIR.vonkarman_constant,
        )

    def st4_spectral_grid(self, spectrum: FrequencyDirectionSpectrum):
        return _st4_spectral_grid(
            spectrum.radian_frequency.values,
            spectrum.radian_direction.values,
            spectrum.frequency_step.values,
            spectrum.direction_step.values,
        )

    def rate(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:

        wind = (speed.values, direction.values, wind_speed_input_type)
        wind_input = _st4_wind_generation(
            spectrum.variance_density.values,
            wind=wind,
            depth=spectrum.depth.values,
            roughness_length=roughness_length.values,
            st4_spectral_grid=self.st4_spectral_grid(spectrum),
            st4_parameters=self.parameters,
        )
        return DataArray(data=wind_input, dims=spectrum.dims, coords=spectrum.coords())

    def bulk_rate(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
        memoized=None,
    ) -> DataArray:
        wind = (speed.values, direction.values, wind_speed_input_type)
        wind_input = _st4_bulk_wind_generation(
            spectrum.variance_density.values,
            wind,
            spectrum.depth.values,
            roughness_length.values,
            self.st4_spectral_grid(spectrum),
            self.parameters,
        )
        return DataArray(
            data=wind_input,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

    def roughness(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length_guess: DataArray = None,
        wind_speed_input_type: TWindInputType = "u10",
    ) -> DataArray:
        if roughness_length_guess is None:
            roughness_length_guess = zeros_like(speed) - 1

        wind = (speed.values, direction.values, wind_speed_input_type)

        rougness = _st4_roughness_estimate(
            guess=roughness_length_guess.values,
            variance_density=spectrum.variance_density.values,
            wind=wind,
            depth=spectrum.depth.values,
            st4_spectral_grid=self.st4_spectral_grid(spectrum),
            st4_parameters=self.parameters,
        )
        return DataArray(
            data=rougness,
            dims=spectrum.dims_space_time,
            coords=spectrum.coords_space_time,
        )

    def u10_from_bulk_rate(
        self,
        bulk_rate: DataArray,
        guess_u10: DataArray,
        guess_direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
    ) -> DataArray:
        """

        :param bulk_rate:
        :param guess_u10:
        :param guess_direction:
        :param spectrum:
        :return:
        """
        disable = spectrum.number_of_spectra < 100
        with ProgressBar(
            total=spectrum.number_of_spectra,
            disable=disable,
            desc="Estimating U10 from ST4 Source term",
        ) as progress_bar:
            u10 = _st4_u10_from_bulk_rate(
                bulk_rate.values,
                spectrum.variance_density.values,
                guess_u10=guess_u10.values,
                guess_direction=guess_direction.values,
                depth=spectrum.depth.values,
                st4_spectral_grid=self.st4_spectral_grid(spectrum),
                st4_parameters=self.parameters,
                progress_bar=progress_bar,
            )
            return DataArray(
                data=u10,
                dims=spectrum.dims_space_time,
                coords=spectrum.coords_space_time,
            )


# ----------------------------------------------------------------------------------------------------------------------
# ST4 Point wise implementation functions (apply to a single spatial point)
# ----------------------------------------------------------------------------------------------------------------------


@njit(cache=True)
def _st4_wind_generation_point(
    variance_density, wind, depth, roughness_length, st4_spectral_grid, st4_parameters
):
    """
    Implememtation of the st4 wind growth rate term

    :param variance_density:
    :param wind:
    :param depth:
    :param roughness_length:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """

    # Get the spectral size
    number_of_frequencies, number_of_directions = variance_density.shape

    # Allocate the output variable
    wind_source = empty((number_of_frequencies, number_of_directions))

    # Unpack variables passed in dictionaries/tuples for ease of reference and some slight speed benifits (avoid dictionary
    # lookup in loops).
    vonkarman_constant = st4_parameters["vonkarman_constant"]
    growth_parameter_betamax = st4_parameters["growth_parameter_betamax"]
    air_density = st4_parameters["air_density"]
    water_density = st4_parameters["water_density"]
    wave_age_tuning_parameter = st4_parameters["wave_age_tuning_parameter"]
    radian_frequency = st4_spectral_grid["radian_frequency"]
    radian_direction = st4_spectral_grid["radian_direction"]
    wind_forcing, wind_direction_degrees, wind_forcing_type = wind

    # Preprocess winds to the correct conventions (U10->friction velocity if need be, wind direction degrees->radians)
    if wind_forcing_type == "u10":
        friction_velocity = (
            wind_forcing * vonkarman_constant / log(10 / roughness_length)
        )

    elif wind_forcing_type in ["ustar", "friction_velocity"]:
        friction_velocity = wind_forcing

    else:
        raise ValueError("unknown wind forcing")
    wind_direction_radian = wind_direction_degrees * pi / 180

    # precalculate the constant multiplication.
    constant_factor = (
        growth_parameter_betamax / vonkarman_constant**2 * air_density / water_density
    )

    # Loop over all frequencies/directions
    for frequency_index in range(number_of_frequencies):

        # Since wavenumber and wavespeed only depend on frequency we calculate those outside of the direction loop.
        constant_frequency_factor = constant_factor * radian_frequency[frequency_index]
        wavenumber = inverse_intrinsic_dispersion_relation(
            radian_frequency[frequency_index], depth
        )[0]
        wavespeed = radian_frequency[frequency_index] / wavenumber

        for direction_index in range(number_of_directions):
            # Calculate the directional factor in the wind input formulation
            W = (
                friction_velocity
                / wavespeed
                * cos(radian_direction[direction_index] - wind_direction_radian)
            )
            if W > 0:
                # If the wave direction has an along wind component we continue.
                effective_wave_age = log(
                    wavenumber * roughness_length
                ) + vonkarman_constant / (W + wave_age_tuning_parameter)
                if effective_wave_age > 0:
                    effective_wave_age = 0

                growth_rate = (
                    constant_frequency_factor
                    * exp(effective_wave_age)
                    * effective_wave_age**4
                    * W**2
                )
            else:
                growth_rate = 0.0

            wind_source[frequency_index, direction_index] = (
                growth_rate * variance_density[frequency_index, direction_index]
            )
    return wind_source


@njit(cache=True)
def _st4_wave_supported_stress_point(
    wind_input: NDArray, depth, st4_spectral_grid, st4_parameters
) -> Tuple[NDArray, NDArray]:
    """
    :param wind_input:
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """

    number_of_frequencies, number_of_directions = wind_input.shape

    stress_east = 0.0
    stress_north = 0.0
    radian_frequency = st4_spectral_grid["radian_frequency"]
    radian_direction = st4_spectral_grid["radian_direction"]
    frequency_step = st4_spectral_grid["frequency_step"]
    direction_step = st4_spectral_grid["direction_step"]

    wavenumber = inverse_intrinsic_dispersion_relation(radian_frequency, depth)
    common_factor = (
        st4_parameters["gravitational_acceleration"] * st4_parameters["water_density"]
    )
    for frequency_index in range(number_of_frequencies):
        inverse_wave_speed = (
            wavenumber[frequency_index] / radian_frequency[frequency_index]
        )
        for direction_index in range(number_of_directions):
            stress_east += (
                cos(radian_direction[direction_index])
                * wind_input[frequency_index, direction_index]
                * inverse_wave_speed
                * frequency_step[frequency_index]
                * direction_step[direction_index]
            )

            stress_north += (
                sin(radian_direction[direction_index])
                * wind_input[frequency_index, direction_index]
                * inverse_wave_speed
                * frequency_step[frequency_index]
                * direction_step[direction_index]
            )
    wave_stress_magnitude = sqrt(stress_north**2 + stress_east**2) * common_factor
    wave_stress_direction = (arctan2(stress_north, stress_east) * 180 / pi) % 360

    return wave_stress_magnitude, wave_stress_direction


@njit(cache=True)
def _st4_charnock_relation_point(friction_velocity, st4_parameters):
    """
    Charnock relation
    :param friction_velocity:
    :param st4_parameters:
    :return:
    """
    roughness_length = (
        friction_velocity**2
        / st4_parameters["gravitational_acceleration"]
        * st4_parameters["charnock_constant"]
    )
    if roughness_length > st4_parameters["charnock_maximum_roughness"]:
        roughness_length = st4_parameters["charnock_maximum_roughness"]
    return roughness_length


@njit()
def _st4_roughness_estimate_point(
    guess, variance_density, wind, depth, st4_spectral_grid, st4_parameters
):
    """

    :param guess:
    :param variance_density:
    :param wind:
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """

    vonkarman_constant = st4_parameters["vonkarman_constant"]
    if guess < 0:
        if wind[2] == "u10":
            drag_coeficient_wu = (0.8 + 0.065 * wind[0]) / 1000
            guess = 10 / exp(vonkarman_constant / sqrt(drag_coeficient_wu))

        elif wind[2] in ["ustar", "friction_velocity"]:
            guess = _st4_charnock_relation_point(wind[0], st4_parameters)
        else:
            raise ValueError("unknown wind forcing")

    if any(isnan(variance_density)):
        return nan

    if wind[0] == 0.0:
        return nan

    function_arguments = (
        variance_density,
        wind,
        depth,
        st4_spectral_grid,
        st4_parameters,
    )
    return numba_fixed_point_iteration(
        _roughness_iteration_function, guess, function_arguments, bounds=(0, inf)
    )


@njit()
def _st4_u10_from_bulk_rate_point(
    bulk_rate,
    variance_density,
    guess_u10,
    guess_direction,
    depth,
    st4_spectral_grid,
    st4_parameters,
):
    """

    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param guess_direction:
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """
    wind_guess = (guess_u10, guess_direction, "u10")
    roughness_memory = [-1.0]
    args = (
        roughness_memory,
        variance_density,
        wind_guess,
        depth,
        st4_spectral_grid,
        st4_parameters,
        bulk_rate,
    )
    if bulk_rate == 0.0:
        return 0.0

    # We are tryomg to solve for U10; ~ accuracy of 0.01m/s is sufficient. We do not really care about relative accuracy
    # - for low winds (<1m/s) answers are really inaccurate anyway
    atol = 1.0e-2
    rtol = 1.0
    numerical_stepsize = 1e-3
    return numba_newton_raphson(
        _u10_iteration_function,
        guess_u10,
        args,
        (0, inf),
        atol=atol,
        rtol=rtol,
        numerical_stepsize=numerical_stepsize,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Apply to all spatial points
# ----------------------------------------------------------------------------------------------------------------------


@njit(cache=True)
def _st4_wind_generation(
    variance_density, wind, depth, roughness_length, st4_spectral_grid, st4_parameters
) -> NDArray:
    (
        number_of_points,
        number_of_frequencies,
        number_of_directions,
    ) = variance_density.shape

    generation = empty((number_of_points, number_of_frequencies, number_of_directions))
    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        wind_generation = _st4_wind_generation_point(
            variance_density[point_index, :, :],
            wind_at_point,
            depth[point_index],
            roughness_length[point_index],
            st4_spectral_grid,
            st4_parameters,
        )
        generation[point_index, :, :] = wind_generation
    return generation


@njit(cache=True)
def _st4_bulk_wind_generation(
    variance_density, wind, depth, roughness_length, st4_spectral_grid, st4_parameters
) -> NDArray:
    number_of_points = variance_density.shape[0]
    generation = empty((number_of_points))

    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        wind_generation = _st4_wind_generation_point(
            variance_density=variance_density[point_index, :, :],
            wind=wind_at_point,
            depth=depth[point_index],
            roughness_length=roughness_length[point_index],
            st4_spectral_grid=st4_spectral_grid,
            st4_parameters=st4_parameters,
        )
        generation[point_index] = numba_integrate_spectral_data(
            wind_generation, st4_spectral_grid
        )
    return generation


@njit()
def _st4_u10_from_bulk_rate(
    bulk_rate,
    variance_density,
    guess_u10,
    guess_direction,
    depth,
    st4_parameters,
    st4_spectral_grid,
    progress_bar=None,
) -> NDArray:
    """

    :param bulk_rate:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param guess_u10:
    :param guess_direction:
    :param depth:
    :param st4_parameters:
    :param st4_spectral_grid:
    :param progress_bar:
    :return:
    """
    number_of_points = variance_density.shape[0]
    u10 = empty((number_of_points))
    for point_index in range(number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        u10[point_index] = _st4_u10_from_bulk_rate_point(
            bulk_rate[point_index],
            variance_density[point_index, :, :],
            guess_u10[point_index],
            guess_direction[point_index],
            depth[point_index],
            st4_spectral_grid=st4_spectral_grid,
            st4_parameters=st4_parameters,
        )
    return u10


@njit()
def _st4_roughness_estimate(
    guess, variance_density, wind, depth, st4_spectral_grid, st4_parameters
) -> NDArray:
    """
    :param guess:
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """
    number_of_points = variance_density.shape[0]
    roughness = empty((number_of_points))

    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])

        if isnan(wind[0][point_index]):
            roughness[point_index] = nan
        else:
            roughness[point_index] = _st4_roughness_estimate_point(
                guess=guess[point_index],
                variance_density=variance_density[point_index, :, :],
                wind=wind_at_point,
                depth=depth[point_index],
                st4_spectral_grid=st4_spectral_grid,
                st4_parameters=st4_parameters,
            )

    return roughness


@njit(cache=True)
def _st4_wave_supported_stress(
    variance_density, wind, depth, roughness_length, st4_spectral_grid, st4_parameters
) -> Tuple[NDArray, NDArray]:
    """
    Calculate the wave supported wind stress.

    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param roughness_length: Surface roughness length in meter.
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """
    number_of_points = variance_density.shape[0]
    magnitude = empty((number_of_points))
    direction = empty((number_of_points))
    for point_index in range(number_of_points):
        wind_at_point = (wind[0][point_index], wind[1][point_index], wind[2])
        wind_generation = _st4_wind_generation_point(
            variance_density=variance_density[point_index, :, :],
            wind=wind_at_point,
            depth=depth[point_index],
            roughness_length=roughness_length[point_index],
            st4_spectral_grid=st4_spectral_grid,
            st4_parameters=st4_parameters,
        )

        (
            magnitude[point_index],
            direction[point_index],
        ) = _st4_wave_supported_stress_point(
            wind_generation, depth[point_index], st4_spectral_grid, st4_parameters
        )
    return magnitude, direction


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------
@njit()
def _st4_parameters(
    air_density,
    water_density,
    wave_age_tuning_parameter=0.00,
    growth_parameter_betamax=1.52,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
    charnock_maximum_roughness=0.0015,
    charnock_constant=0.018,
    vonkarman_constant=0.4,
):
    out = {
        "air_density": air_density,
        "water_density": water_density,
        "wave_age_tuning_parameter": wave_age_tuning_parameter,
        "growth_parameter_betamax": growth_parameter_betamax,
        "gravitational_acceleration": gravitational_acceleration,
        "charnock_maximum_roughness": charnock_maximum_roughness,
        "charnock_constant": charnock_constant,
        "vonkarman_constant": vonkarman_constant,
    }
    return out


@njit()
def _st4_spectral_grid(
    radian_frequency, radian_direction, frequency_step, direction_step
):
    return {
        "radian_frequency": radian_frequency,
        "radian_direction": radian_direction,
        "frequency_step": frequency_step,
        "direction_step": direction_step,
    }


@njit()
def _roughness_iteration_function(
    roughness_length, variance_density, wind, depth, st4_spectral_grid, st4_parameters
):
    """
    The surface roughness is defined implicitly. We use a fixed-point iteration step to solve for the roughness. This
    is the iteration function used in the fixed point iteration.

    :param roughness_length: Surface roughness length in meter.
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind: wind input tuple (magnitude, direction_degrees, type_string)
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :return:
    """
    vonkarman_constant = st4_parameters["vonkarman_constant"]
    if wind[2] == "u10":
        friction_velocity = wind[0] * vonkarman_constant / log(10 / roughness_length)
    else:
        friction_velocity = wind[0]

    # Get the wind input source term values
    generation = _st4_wind_generation_point(
        variance_density,
        wind,
        depth,
        roughness_length,
        st4_spectral_grid,
        st4_parameters,
    )

    # Calculate the stress
    wave_supported_stress, _ = _st4_wave_supported_stress_point(
        generation, depth, st4_spectral_grid, st4_parameters
    )

    # Given the stress ratio, correct the Charnock roughness for the presence of waves.
    total_stress = st4_parameters["air_density"] * friction_velocity**2
    stress_ratio = wave_supported_stress / total_stress

    if stress_ratio > 0.9:
        stress_ratio = 0.9

    charnock_roughness = _st4_charnock_relation_point(friction_velocity, st4_parameters)
    return charnock_roughness / sqrt(1 - stress_ratio)


@njit()
def _u10_iteration_function(
    U10,
    memory_list,
    variance_density,
    wind_guess,
    depth,
    st4_spectral_grid,
    st4_parameters,
    bulk_rate,
):
    """
    To find the 10 meter wind that generates a certain bulk input we need to solve the inverse function for the
    wind input term. We define a function

            F(U10) = wind_bulk_rate(u10) - specified_bulk_rate

    and use a zero finder to solve for

            F(U10) = 0.

    This function defines the function F. The zero finder is responsible for calling it with the correct trailing
    arguments.

    :param wind_forcing: Wind input U10 we want to solve for.
    :param variance_density: The wave spectrum in m**2/Hz/deg
    :param wind_guess:
    :param depth:
    :param st4_spectral_grid:
    :param st4_parameters:
    :param bulk_rate:
    :return:
    """
    if U10 == 0.0:
        return -bulk_rate

    wind = (U10, wind_guess[1], wind_guess[2])

    # Estimate the rougness length
    roughness_length = _st4_roughness_estimate_point(
        memory_list[0], variance_density, wind, depth, st4_spectral_grid, st4_parameters
    )
    memory_list[0] = roughness_length

    # Calculate the wind input source term values
    generation = _st4_wind_generation_point(
        variance_density,
        wind,
        depth,
        roughness_length,
        st4_spectral_grid,
        st4_parameters,
    )

    # Integrate the input and return the difference of the current guess with the desired bulk rate
    return numba_integrate_spectral_data(generation, st4_spectral_grid) - bulk_rate
