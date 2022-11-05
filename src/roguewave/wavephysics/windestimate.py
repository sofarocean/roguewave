"""
Contents: Wind Estimator

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import typing
import numpy
from typing import Literal
from roguewave import FrequencySpectrum, FrequencyDirectionSpectrum, WaveSpectrum
from roguewave.wavephysics.balance import SourceTermBalance
from xarray import Dataset
from roguewave.wavephysics.fluidproperties import AIR
from roguewave.wavephysics.momentumflux import (
    RoughnessLength,
    create_roughness_length_estimator,
)
from roguewave.tools.solvers import newton_raphson, Configuration

_methods = Literal["peak", "mean"]
_charnock_parametrization = Literal["constant", "voermans15", "voermans16"]
_direction_convention = Literal[
    "coming_from_clockwise_north", "going_to_counter_clockwise_east"
]


def friction_velocity(
    spectrum: FrequencySpectrum,
    method: _methods = "peak",
    fmax=0.5,
    power=4,
    directional_spreading_constant=2.5,
    beta=0.012,
    grav=9.81,
    number_of_bins=20,
) -> Dataset:
    """

    :param spectrum:
    :param method:
    :param fmax:
    :param power:
    :param directional_spreading_constant:
    :param beta:
    :param grav:
    :param number_of_bins:
    :return:
    """
    e, a1, b1 = equilibrium_range_values(
        spectrum,
        method=method,
        fmax=fmax,
        power=power,
        number_of_bins=number_of_bins,
    )

    emean = 8.0 * numpy.pi**3 * e

    # Get friction velocity from spectrum
    friction_velocity_estimate = (
        emean / grav / directional_spreading_constant / beta / 4
    )

    # Estimate direction from tail
    direction = (180.0 / numpy.pi * numpy.arctan2(b1, a1)) % 360
    coords = {x: spectrum.dataset[x].values for x in spectrum.dims_space_time}
    return Dataset(
        data_vars={
            "friction_velocity": (spectrum.dims_space_time, friction_velocity_estimate),
            "direction": (spectrum.dims_space_time, direction),
        },
        coords=coords,
    )


def estimate_u10_from_spectrum(
    spectrum: WaveSpectrum,
    method: _methods = "peak",
    fmax=0.5,
    power=4,
    directional_spreading_constant=2.5,
    phillips_constant_beta=0.012,
    vonkarman_constant=0.4,
    grav=9.81,
    number_of_bins=20,
    roughness_parametrization: RoughnessLength = None,
    direction_convention: _direction_convention = "going_to_counter_clockwise_east",
) -> Dataset:
    #
    # =========================================================================
    # Required Input
    # =========================================================================
    #
    # f              :: frequencies (in Hz)
    # E              :: Variance densities (in m^2 / Hz )
    #
    # =========================================================================
    # Output
    # =========================================================================
    #
    # U10            :: in m/s
    # Direction      :: in degrees clockwise from North (where wind is *coming from)
    #
    # =========================================================================
    # Named Keywords (parameters to inversion algorithm)
    # =========================================================================
    # Npower = 4     :: exponent for the fitted f^-N spectral tail
    # I      = 2.5   :: Philips Directional Constant
    # beta   = 0.012 :: Equilibrium Constant
    # Kapppa = 0.4   :: Von Karman constant
    # Alpha  = 0.012 :: Constant in estimating z0 from u* in Charnock relation
    # grav   = 9.81  :: Gravitational acceleration
    #
    # =========================================================================
    # Algorithm
    # =========================================================================
    #
    # 1) Find the part of the spectrum that best fits a f^-4 shape
    # 2) Estimate the Phillips equilibrium level "Emean" over that range
    # 3) Use Emean to estimate Wind speed (using Charnock and LogLaw)
    # 4) Calculate mean direction over equilibrium range

    if roughness_parametrization is None:
        roughness_parametrization = create_roughness_length_estimator()

    if isinstance(spectrum, FrequencyDirectionSpectrum):
        spectrum_1d = spectrum.as_frequency_spectrum()
    else:
        spectrum_1d = spectrum

    # Get friction velocity from spectrum
    dataset = friction_velocity(
        spectrum_1d,
        method,
        fmax,
        power,
        directional_spreading_constant,
        phillips_constant_beta,
        grav,
        number_of_bins,
    )
    # Find z0 from Charnock Relation
    z0 = roughness_parametrization.roughness(
        dataset.friction_velocity, spectrum, AIR, dataset.direction
    )

    if direction_convention == "coming_from_clockwise_north":
        dataset["direction"] = (270.0 - dataset["direction"]) % 360
    elif direction_convention == "going_to_counter_clockwise_east":
        pass
    else:
        raise ValueError(f"Unknown direectional convention: {direction_convention}")

    # Get the wind speed at U10 from loglaw
    return dataset.assign(
        {"u10": dataset.friction_velocity / vonkarman_constant * numpy.log(10.0 / z0)}
    )


def equilibrium_range_values(
    spectrum: FrequencySpectrum,
    method: _methods,
    fmax=1.25,
    power=4,
    number_of_bins=20,
) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """

    :param spectrum:
    :param method:
    :param fmax:
    :param power:
    :param number_of_bins:
    :return:
    """

    if method == "peak":
        scaled_spec = spectrum.variance_density * spectrum.frequency**power
        indices = scaled_spec.argmax(dim="frequency")
        a1 = spectrum.a1.isel({"frequency": indices}).values
        b1 = spectrum.b1.isel({"frequency": indices}).values
        e = scaled_spec.isel({"frequency": indices}).values
        return e, a1, b1

    elif method == "mean":
        scaled_spec = spectrum.variance_density * spectrum.frequency**power

        # Find fmin/fmax
        fmin = 0
        i_min = numpy.argmin(numpy.abs(spectrum.frequency.values - fmin), axis=-1)
        i_max = numpy.argmin(numpy.abs(spectrum.frequency.values - fmax), axis=-1)
        nf = spectrum.number_of_frequencies

        i_max = i_max + 1 - number_of_bins
        i_max = numpy.max((i_min + 1, i_max))
        i_max = numpy.min((i_max, nf - number_of_bins))

        i_counter = 0
        shape = list(spectrum.shape())
        shape[-1] = i_max - i_min
        variance = numpy.zeros(shape) + numpy.inf

        #
        # Calculate the variance with respect to a running average mean of numberOfBins
        #
        for iFreq in range(i_min, i_max):
            #
            # Ensure we do not go out of bounds
            iiu = numpy.min((iFreq + number_of_bins, nf))

            # Ensure there are no 0 contributions (essentially no data)
            m = scaled_spec[..., iFreq:iiu].mean(dim="frequency")
            variance[..., i_counter] = ((scaled_spec[..., iFreq:iiu] - m) ** 2).mean(
                dim="frequency"
            ) / (m**2)
            i_counter = i_counter + 1
            #
        #
        i_min_variance = numpy.argmin(variance, axis=-1) + i_min

        e = numpy.zeros(i_min_variance.shape)
        a1 = numpy.zeros(i_min_variance.shape)
        b1 = numpy.zeros(i_min_variance.shape)

        index = numpy.array(list(range(0, spectrum.number_of_spectra)))
        index = numpy.unravel_index(index, i_min_variance.shape)
        i_min_variance = i_min_variance.flatten()

        for ii in range(0, number_of_bins):
            jj = numpy.clip(i_min_variance + ii, a_min=0, a_max=nf - 1 - number_of_bins)

            indexer = tuple([ind for ind in index] + [jj])

            e[index] += scaled_spec.values[indexer]
            a1[index] += spectrum.a1.values[indexer]
            b1[index] += spectrum.b1.values[indexer]
        fac = 1 / number_of_bins
        e *= fac
        a1 *= fac
        b1 *= fac

        return e, a1, b1


def estimate_u10_from_source_terms(
    spectrum: FrequencyDirectionSpectrum,
    balance: SourceTermBalance,
    roughness: RoughnessLength,
    method="newton",
    solver_configuration: Configuration = None,
    **kwargs,
) -> Dataset:
    if method == "newton":
        return _estimate_u10_from_source_terms_newton(
            spectrum, balance, roughness, solver_configuration, **kwargs
        )
    else:
        raise ValueError("unknown method")


def _estimate_u10_from_source_terms_newton(
    spectrum: FrequencyDirectionSpectrum,
    balance: SourceTermBalance,
    roughness: RoughnessLength,
    solver_configuration: Configuration = None,
    **kwargs,
):
    # Estimate the wind direction. Note this is our _final_ estimate of the wind direction.
    memoize = {}
    wind_direction = balance.dissipation.mean_direction_degrees(
        spectrum, memoize=memoize
    )
    dissipation_bulk = balance.dissipation.bulk_rate(spectrum, memoize=memoize)

    if solver_configuration is None:
        solver_configuration = Configuration(atol=1.0e-2, rtol=1.0e-3, use_numba=True)

    guess = None
    # Define the iteration function
    memoize = {}

    def func(u10):
        nonlocal guess

        roughness_length = balance.generation.roughness(
            u10, wind_direction, spectrum, roughness_length_guess=guess
        )
        guess = roughness_length
        return (
            balance.generation.bulk_rate(
                u10, wind_direction, spectrum, roughness_length
            )
            + dissipation_bulk
        )

    # Our first guess is the wind estimate based on the peak equilibrium range approximation
    guess = estimate_u10_from_spectrum(
        spectrum, "peak", direction_convention="going_to_counter_clockwise_east"
    )["u10"]

    if solver_configuration.use_numba:
        u10_estimate = balance.generation.u10_from_bulk_rate(
            -dissipation_bulk, guess, wind_direction, spectrum
        )
    else:
        u10_estimate = newton_raphson(
            func, guess, bounds=(0, numpy.inf), configuration=solver_configuration
        )

    dataset = Dataset()
    return dataset.assign({"u10": u10_estimate, "direction": wind_direction})
