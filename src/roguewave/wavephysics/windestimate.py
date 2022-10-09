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
from xarray import Dataset, ones_like, DataArray

_methods = Literal["peak", "mean"]
_charnock_parametrization = Literal["constant", "voermans15", "voermans16"]


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
    dir = (270.0 - 180.0 / numpy.pi * numpy.arctan2(b1, a1)) % 360

    coords = {x: spectrum.dataset[x].values for x in spectrum.dims_space_time}
    return Dataset(
        data_vars={
            "friction_velocity": (spectrum.dims_space_time, Ustar),
            "direction_degrees": (spectrum.dims_space_time, dir),
        },
        coords=coords,
    )


def U10(
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
        for ii in range(0, number_of_bins):
            e += scaled_spec.values[index, iMinVariance + ii]
            a1 += spectrum.a1.values[index, iMinVariance + ii]
            b1 += spectrum.b1.values[index, iMinVariance + ii]
        fac = 1 / number_of_bins
        e *= fac
        a1 *= fac
        b1 *= fac

        return e, a1, b1
