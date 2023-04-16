import roguewave.wavephysics.balance.jb23_tail_stress as jb23
from roguewave.wavephysics.balance.jb23_wind_input import JB23WindInput
from roguewave.wavephysics.roughness import (
    charnock_roughness_length_from_u10,
    drag_coefficient_charnock,
)
from roguewave.wavespectra.parametric import (
    create_parametric_frequency_direction_spectrum,
)
from roguewave import FrequencyDirectionSpectrum, concatenate_spectra
import matplotlib.pyplot as plt
import numpy as np
from xarray import DataArray

_input = JB23WindInput()
parameters = _input.parameters


def roughness_and_friction_vel(u10):
    U10 = np.array([u10])
    roughness_length = charnock_roughness_length_from_u10(U10)
    drag = drag_coefficient_charnock(U10)
    friction_velocity = np.sqrt(drag) * U10
    return roughness_length.values[0], friction_velocity.values[0]


def alpha_from_wave_age(wave_age):
    return np.tanh(0.24 * wave_age ** (-1) / 0.031) * 0.031
    # return np.tanh(0.57 * wave_age ** (-3/2) / 0.031) * 0.031


def m0_from_wave_age(wave_age, peak_frequency):
    alpha = alpha_from_wave_age(wave_age)
    g = 9.81
    gamma = 3.3
    return (
        alpha
        * g**2
        / (2 * np.pi) ** 4
        / peak_frequency**4
        * (0.06533 * gamma**0.8015 + 0.13467)
    )


def spectrum_from_u10_and_wave_age_guess(U10, wave_age) -> FrequencyDirectionSpectrum:
    roughness, friction_vel = roughness_and_friction_vel(U10)
    peak_celerity = wave_age * friction_vel
    peak_frequency = 9.81 / peak_celerity / 2 / np.pi
    m0 = m0_from_wave_age(wave_age, peak_frequency)

    resolved_frequencies = np.linspace(0.01, 0.8, 80)
    hs = 4 * np.sqrt(m0)
    return create_parametric_frequency_direction_spectrum(
        resolved_frequencies, peak_frequency, hs
    )


def gen_saturation_spectrum(u10, wave_age=25):
    roughness_length, friction_velocity = roughness_and_friction_vel(u10)

    friction_velocity = friction_velocity
    spectrum = spectrum_from_u10_and_wave_age_guess(u10, wave_age)
    spec1d = spectrum.as_frequency_spectrum()

    energy_at_starting_wavenumber = (
        spec1d.group_velocity.values[-1]
        * spec1d.variance_density.values[-1]
        / 2
        / np.pi
    )
    starting_wavenumber = spec1d.wavenumber.values[-1]

    wavenumbers = jb23.wavenumber_grid(
        starting_wavenumber, roughness_length, friction_velocity, parameters
    )

    return wavenumbers, jb23.saturation_spectrum_parametrization(
        wavenumbers,
        energy_at_starting_wavenumber,
        starting_wavenumber,
        friction_velocity,
        parameters,
    )


def t_wavenumber_grid():
    u10 = np.array([10])
    roughness_length = charnock_roughness_length_from_u10(u10)
    drag = drag_coefficient_charnock(u10)
    friction_velocity = np.sqrt(drag) * u10

    friction_velocity = friction_velocity.values[0]
    roughness_length = roughness_length.values[0]

    eq_range_limit = jb23.upper_limit_wavenumber_equilibrium_range(
        friction_velocity, parameters
    )
    three_wave_limit = jb23.three_wave_starting_wavenumber(
        friction_velocity, parameters
    )

    def gen_grid(start, isempty, has_eq, has_cons, has_3w):
        grid = jb23.wavenumber_grid(
            start, roughness_length, friction_velocity, parameters
        )
        # Assert that this is empty (or not)
        assert isempty == (grid[0] < 0)
        # Assert that this has eq range
        assert has_eq == (np.any(grid < eq_range_limit) and np.all(grid > 0))
        assert has_cons == (
            np.any((grid < three_wave_limit) & (grid >= eq_range_limit))
            and np.all(grid > 0)
        )
        assert has_3w == (np.any(grid > three_wave_limit) and np.all(grid > 0))

        return grid

    grid = gen_grid(eq_range_limit * 0.1, False, True, True, True)
    grid = gen_grid(eq_range_limit * 1.1, False, False, True, True)
    grid = gen_grid(three_wave_limit * 1.1, False, False, False, True)
    grid = gen_grid(100000, True, False, False, False)

    print(grid / eq_range_limit)


def t_log_bounds():
    u10_vec = np.linspace(0, 100, 1001)

    for u10val in u10_vec:
        u10 = np.array([u10val])
        roughness_length = charnock_roughness_length_from_u10(u10)
        drag = drag_coefficient_charnock(u10)
        friction_velocity = np.sqrt(drag) * u10

        friction_velocity = friction_velocity.values[0]
        roughness_length = roughness_length.values[0]

        bounds = jb23.log_bounds_wavenumber(
            roughness_length, friction_velocity, parameters
        )

        if np.isfinite(bounds[1]):
            lower_bound_miles = jb23.miles_mu_cutoff(
                bounds[0], roughness_length, friction_velocity, parameters
            )
            upper_bound_miles = jb23.miles_mu_cutoff(
                bounds[1], roughness_length, friction_velocity, parameters
            )
            assert np.abs(lower_bound_miles) < 1e-4
            assert np.abs(upper_bound_miles) < 1e-4


def t_log_bounds_hard():
    u10_vec = [5]

    for u10val in u10_vec:
        friction_velocity = 0.3856067597167886
        roughness_length = 0.05590810182512223

        bounds = jb23.log_bounds_wavenumber(
            roughness_length, friction_velocity, parameters
        )
        print(bounds)


def t_sat_spec():
    u10 = 15

    k, sat_spec = gen_saturation_spectrum(u10, wave_age=30)
    en_spec = sat_spec * k ** (-3)
    roughness, fric_vel = roughness_and_friction_vel(u10)
    wind_input = jb23.wind_input_tail(k, roughness, fric_vel, en_spec, parameters)

    plt.plot(k, sat_spec / sat_spec[0])
    plt.xscale("log")
    plt.figure()

    plt.plot(k, wind_input * k ** (1 / 2))

    plt.yscale("log")

    plt.show()


def t_stress():
    u10 = 20
    spectrum = spectrum_from_u10_and_wave_age_guess(U10=u10, wave_age=50)
    spectrum = concatenate_spectra([spectrum])

    roughness = 10 ** np.linspace(-9, -1, 1000)
    ustar = u10 * 0.4 / np.log(10 / roughness)
    speed = DataArray([u10])
    direction = DataArray([0.0])

    stress = np.zeros_like(roughness)
    for index, val in enumerate(roughness):
        r = DataArray(data=[val])
        res = _input.stress(spectrum, speed, direction, r)
        stress[index] = res["stress"]

    r = _input.roughness(speed, direction, spectrum)
    ustart = _input.friction_velocity(spectrum, speed, direction, r)
    d = (ustart / u10) ** 2
    spect1d = spectrum.as_frequency_spectrum()
    y = (
        (np.log(1 / roughness) - np.log(9.81 / ustar**2))
        * spect1d.saturation_spectrum.values[0, -1]
        * 50
    )

    drag = (ustar / u10) ** 2
    plt.plot(drag, stress / ustar**2 / 1.225)
    plt.plot(drag, ustar**2 * 0 + 1, "k--")
    plt.plot(drag, y, "r--")
    plt.plot([d.values[0], d.values[0]], [0, 2], "r--")
    plt.ylim([0, 2])
    # plt.xscale('log')
    plt.show()


def celerity(wavenumber, gravitational_acceleration, surface_tension):
    return np.sqrt(
        gravitational_acceleration / wavenumber + surface_tension * wavenumber
    )


if __name__ == "__main__":
    t_stress()
