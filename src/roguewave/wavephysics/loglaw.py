from typing import Union
from roguewave import WaveSpectrum
from xarray import DataArray, where, ones_like
from numpy.typing import ArrayLike
from .roughnesslength import RoughnessLength
import numpy

VONKARMANCONSTANT = 0.4


def friction_velocity_from_windspeed(
    windspeed: Union[ArrayLike, DataArray],
    spectrum: WaveSpectrum,
    roughness_length: RoughnessLength,
    elevation=10,
    vonkarmanconstant=VONKARMANCONSTANT,
    max_iter=100,
    rtol=1e-7,
    atol=1e-7,
):
    def func(ustar):
        return windspeed - ustar / vonkarmanconstant * numpy.log(
            elevation / roughness_length.z0(ustar, spectrum)
        )

    def derivative_func(ustar):
        return -numpy.log(
            elevation / roughness_length.z0(ustar, spectrum)
        ) / vonkarmanconstant + ustar / vonkarmanconstant / roughness_length.z0(
            ustar, spectrum
        ) * roughness_length.derivative_to_ustar(
            ustar, spectrum
        )

    initial_guess = ones_like(windspeed)
    f = func(initial_guess)
    for ii in range(0, max_iter):
        delta = -f / derivative_func(initial_guess)
        updated_guess = initial_guess + delta

        # Ensure the new value of the friction velocity is larger than 0; if not, halve the current guess.
        initial_guess = where(updated_guess > 0, updated_guess, 0.5 * initial_guess)
        f = func(initial_guess)
        if numpy.all((numpy.abs(f) < atol) & (numpy.abs(f) / windspeed < rtol)):
            break
    else:
        raise ValueError(f"No convervence after {max_iter} iterations")
    return updated_guess


def loglaw(
    friction_velocity,
    elevation,
    spectrum: WaveSpectrum,
    roughness_length: RoughnessLength,
    vonkarmanconstant=VONKARMANCONSTANT,
):
    return (
        friction_velocity
        / vonkarmanconstant
        * numpy.log(elevation / roughness_length.z0(friction_velocity, spectrum))
    )
