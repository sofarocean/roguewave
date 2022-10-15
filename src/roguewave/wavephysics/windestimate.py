"""
This file is part of pysofar: A client for interfacing with Sofar Oceans Spotter API

Contents: Wind Estimater

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import typing
import numpy
from typing import Literal
from roguewave import FrequencySpectrum
from roguewave.wavephysics.balance import SourceTermBalance
from xarray import Dataset, ones_like, DataArray, where
from multiprocessing import get_context, cpu_count
from copy import deepcopy
from scipy.optimize import brentq
from tqdm import tqdm

_methods = Literal["peak", "mean"]
_charnock_parametrization = Literal["constant", "voermans15", "voermans16"]
_direction_convention = Literal[
    "coming_from_clockwise_north", "going_to_counter_clockwise_east"
]


def friction_velocity(
    spectrum: FrequencySpectrum,
    method: _methods = "peak",
    fmin=-1.0,
    fmax=0.5,
    power=4,
    directional_spreading_constant=2.5,
    beta=0.012,
    grav=9.81,
    numberOfBins=20,
    direction_convention: _direction_convention = "coming_from_clockwise_north",
) -> Dataset:
    e, a1, b1 = equilibrium_range_values(
        spectrum,
        method=method,
        fmin=fmin,
        fmax=fmax,
        power=power,
        number_of_bins=numberOfBins,
    )

    Emean = 8.0 * numpy.pi**3 * e

    # Get friction velocity from spectrum
    Ustar = Emean / grav / directional_spreading_constant / beta / 4

    # Convert directions to where the wind is coming from, measured positive clockwise from North

    if direction_convention == "coming_from_clockwise_north":
        dir = (270.0 - 180.0 / numpy.pi * numpy.arctan2(b1, a1)) % 360
    elif direction_convention == "going_to_counter_clockwise_east":
        dir = 180.0 / numpy.pi * numpy.arctan2(b1, a1)
    else:
        raise ValueError(f"Unknown direectional convention: {direction_convention}")

    coords = {x: spectrum.dataset[x].values for x in spectrum.dims_space_time}
    return Dataset(
        data_vars={
            "friction_velocity": (spectrum.dims_space_time, Ustar),
            "direction": (spectrum.dims_space_time, dir),
        },
        coords=coords,
    )


def estimate_u10_from_spectrum(
    spectrum: FrequencySpectrum,
    method: _methods = "peak",
    fmin=-1.0,
    fmax=0.5,
    power=4,
    directional_spreading_constant=2.5,
    phillips_constant_beta=0.012,
    vonkarman_constant=0.4,
    charnock_constant=0.012,
    grav=9.81,
    number_of_bins=20,
    charnock_parametrization: _charnock_parametrization = "constant",
    direction_convention: _direction_convention = "coming_from_clockwise_north",
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

    # Get friction velocity from spectrum
    dataset = friction_velocity(
        spectrum,
        method,
        fmin,
        fmax,
        power,
        directional_spreading_constant,
        phillips_constant_beta,
        grav,
        number_of_bins,
        direction_convention,
    )

    charnock = charnock_constant_estimate(
        spectrum, charnock_constant, dataset.friction_velocity, charnock_parametrization
    )

    # Find z0 from Charnock Relation
    z0 = charnock * dataset.friction_velocity**2 / grav

    # Get the wind speed at U10 from loglaw
    return dataset.assign(
        {"U10": dataset.friction_velocity / vonkarman_constant * numpy.log(10.0 / z0)}
    )


def charnock_constant_estimate(
    spectrum: FrequencySpectrum,
    charnock_constant,
    ustar,
    charnock_parametrization: _charnock_parametrization = "constant",
) -> DataArray:
    hm0 = spectrum.hm0()
    if charnock_parametrization == "constant":
        return charnock_constant * ones_like(hm0)

    elif charnock_parametrization == "voermans15":
        kp = spectrum.peak_wavenumber
        return 0.06 * (hm0 * kp) ** 0.7
    elif charnock_parametrization == "voermans16":
        kp = spectrum.peak_wavenumber
        fp = spectrum.peak_frequency()
        cp = numpy.pi * 2 * fp / kp
        return 0.14 * (ustar / cp) ** 0.61


def equilibrium_range_values(
    spectrum: FrequencySpectrum,
    method: _methods,
    fmin=0.0293,
    fmax=1.25,
    power=4,
    number_of_bins=20,
) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """

    :param spectrum:
    :param method:
    :param fmin:
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
        iMin = numpy.argmin(numpy.abs(spectrum.frequency.values - fmin), axis=-1)
        iMax = numpy.argmin(numpy.abs(spectrum.frequency.values - fmax), axis=-1)
        nf = spectrum.number_of_frequencies

        iMax = iMax + 1 - number_of_bins
        iMax = numpy.max((iMin + 1, iMax))
        iMax = numpy.min((iMax, nf - number_of_bins))

        iCounter = 0
        shape = list(spectrum.shape())
        shape[-1] = iMax - iMin
        Variance = numpy.zeros(shape) + numpy.inf

        #
        # Calculate the variance with respect to a running average mean of numberOfBins
        #
        for iFreq in range(iMin, iMax):
            #
            # Ensure we do not go out of bounds
            iiu = numpy.min((iFreq + number_of_bins, nf))

            # Ensure there are no 0 contributions (essentially no data)
            M = scaled_spec[..., iFreq:iiu].mean(dim="frequency")
            Variance[..., iCounter] = ((scaled_spec[..., iFreq:iiu] - M) ** 2).mean(
                dim="frequency"
            ) / (M**2)
            iCounter = iCounter + 1
            #
        #
        iMinVariance = numpy.argmin(Variance, axis=-1) + iMin

        e = numpy.zeros(iMinVariance.shape)
        a1 = numpy.zeros(iMinVariance.shape)
        b1 = numpy.zeros(iMinVariance.shape)

        index = numpy.array(list(range(0, spectrum.number_of_spectra)))
        index = numpy.unravel_index(index, iMinVariance.shape)
        iMinVariance = iMinVariance.flatten()

        for ii in range(0, number_of_bins):
            jj = numpy.clip(iMinVariance + ii, a_min=0, a_max=nf - 1 - number_of_bins)

            indexer = tuple([ind for ind in index] + [jj])

            e[index] += scaled_spec.values[indexer]
            a1[index] += spectrum.a1.values[indexer]
            b1[index] += spectrum.b1.values[indexer]
        fac = 1 / number_of_bins
        e *= fac
        a1 *= fac
        b1 *= fac

        return e, a1, b1


def _worker(args):
    balance: SourceTermBalance = args[0]
    spec: FrequencySpectrum = args[1]
    direction: DataArray = args[2]

    spec2d = spec.as_frequency_direction_spectrum(36)

    def func(U10):
        return balance.evaluate_bulk_imbalance(U10, direction, spec2d).values[0]

    lower_bound = 0
    upper_bound = 100
    if func(0) * func(100) > 0:
        return numpy.nan
    return brentq(func, a=lower_bound, b=upper_bound)


def estimate_u10_from_source_terms(
    spectrum: FrequencySpectrum, balance: SourceTermBalance, method="newton", **kwargs
) -> DataArray:
    if method == "newton":
        return _estimate_u10_from_source_terms_newton(spectrum, balance, **kwargs)
    elif method == "brentq":
        return _estimate_u10_from_source_terms_brentq(spectrum, balance, **kwargs)
    else:
        raise ValueError("unknown method")


def _estimate_u10_from_source_terms_brentq(
    spectrum: FrequencySpectrum, balance: SourceTermBalance, parallel=False
):
    observed = estimate_u10_from_spectrum(
        spectrum, "peak", direction_convention="going_to_counter_clockwise_east"
    )
    wind_direction = observed["direction"]

    if parallel:
        # We have to force a load from disc if we want to use multiprocessing.
        spectrum.dataset.load()

    number_of_spectra = spectrum.number_of_spectra
    work = []
    for ii in range(number_of_spectra):
        spec = spectrum[slice(ii, ii + 1, 1), :]
        work.append((deepcopy(balance), spec, wind_direction[ii]))
    if parallel:
        with get_context("spawn").Pool(processes=cpu_count()) as pool:
            out = list(tqdm(pool.imap(_worker, work), total=number_of_spectra))
    else:
        out = list(tqdm(map(_worker, work), total=number_of_spectra))

    u10 = DataArray(
        data=numpy.array(out),
        dims=spectrum.dims_space_time,
        coords={x: spectrum.dataset[x].values for x in spectrum.dims_space_time},
    )
    dataset = Dataset()
    return dataset.assign({"U10": u10, "direction": wind_direction})


def _estimate_u10_from_source_terms_newton(
    spectrum: FrequencySpectrum,
    balance: SourceTermBalance,
    atol=1e-3,
    rtol=1e-3,
    max_iter=100,
):
    observed = estimate_u10_from_spectrum(
        spectrum, "peak", direction_convention="going_to_counter_clockwise_east"
    )
    wind_direction = observed["direction"]
    guess = observed["U10"]

    cur_iter = guess

    spec2d = spectrum.as_frequency_direction_spectrum(36)
    dissipation = balance.dissipation.bulk_rate(spec2d)

    def func(u10):
        return balance.generation.bulk_rate(u10, wind_direction, spec2d) + dissipation

    cur_func = func(cur_iter)

    for ii in range(0, max_iter):
        delta = cur_iter * 0.01

        # Note- actually numerically evaluating the derivative is more stable than a
        # newton-raphson iteration
        derivative = (func(cur_iter + delta) - cur_func) / delta / 2

        updated_guess = cur_iter - cur_func / derivative
        updated_guess = where(updated_guess > 0, updated_guess, 0.5 * cur_iter)

        prev_iter = cur_iter
        cur_iter = updated_guess

        cur_func = func(cur_iter)
        delta = (cur_iter - prev_iter) / cur_iter

        msk = numpy.isfinite(cur_func)
        if numpy.all(
            (numpy.abs(cur_func.values[msk]) < atol)
            & (
                numpy.abs(cur_func.values[msk]) / numpy.abs(dissipation.values[msk])
                < rtol
            )
        ):
            break

        if numpy.all(numpy.abs(delta) < rtol):
            break

    else:
        # raise ValueError(f'No convergence after {max_iter} iterations')
        pass

    dataset = Dataset()
    return dataset.assign({"U10": cur_iter, "direction": wind_direction})
