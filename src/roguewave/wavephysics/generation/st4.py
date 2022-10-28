from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from roguewave import FrequencyDirectionSpectrum, SPECTRAL_DIMS
from roguewave.wavephysics.generation import WindGeneration
from numpy import cos, pi, log, exp, tanh, sqrt
from xarray import DataArray, where
from roguewave import integrate_spectral_data
from scipy.special import kelvin


class ST4(WindGeneration):
    def __init__(
        self,
        wave_age_tuning_parameter=0.006,
        growth_parameter_betamax=0.5,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
        swell_dissipation_coefficients=None,
        **kwargs,
    ):
        super(ST4, self).__init__(**kwargs)

        if swell_dissipation_coefficients is None:
            swell_dissipation_coefficients = {
                "s0": 3,
                "s1": 0.8,
                "s2": -0.018,
                "s3": 0.015,
                "rz0": 0.04,
                "laminar_coeficient_cds": 1.2,
                "dimensional_critical_reynolds_number": 2e5,
            }

        self.wave_age_tuning_parameter = wave_age_tuning_parameter
        self.growth_parameter_betamax = growth_parameter_betamax
        self.gravitational_acceleration = gravitational_acceleration
        self.swell_dissipation_coefficients = swell_dissipation_coefficients

    def rate_friction_velocity(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:

        memoized = memoized if memoized is not None else {}
        if "wave_speed" not in memoized:
            memoized["wave_speed"] = spectrum.wave_speed()

        if "wavenumber" not in memoized:
            memoized["wavenumber"] = spectrum.wavenumber

        wind_input = st4_wind_source_term_ustar(
            spectrum.variance_density,
            spectrum.radian_frequency,
            speed,
            direction,
            spectrum.direction,
            memoized["wavenumber"],
            memoized["wave_speed"],
            roughness_length,
            self.wave_age_tuning_parameter,
            self.growth_parameter_betamax,
            water,
            air,
        )
        return wind_input + self.swell_dissipation(
            speed, direction, spectrum, roughness_length, air, water, memoized
        )

    def rate_U10(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length: DataArray = None,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:

        friction_velocity = air.vonkarman_constant * speed / log(10 / roughness_length)
        return self.rate_friction_velocity(
            friction_velocity,
            direction,
            spectrum,
            roughness_length,
            air,
            water,
            memoized,
        )

    def swell_dissipation(
        self,
        speed: DataArray,
        direction: DataArray,
        spectrum: FrequencyDirectionSpectrum,
        roughness_length,
        air: FluidProperties = AIR,
        water: FluidProperties = WATER,
        memoized=None,
    ) -> DataArray:

        memoized = memoized if memoized is not None else {}
        if "wave_speed" not in memoized:
            memoized["wave_speed"] = spectrum.wave_speed()

        if "wavenumber" not in memoized:
            memoized["wavenumber"] = spectrum.wavenumber

        if "significant_wave_height" not in memoized:
            memoized["significant_wave_height"] = spectrum.significant_waveheight

        if "significant_orbital_velocity" not in memoized:
            memoized["significant_orbital_velocity"] = st4_significant_orbital_velocity(
                spectrum.variance_density,
                spectrum.radian_frequency,
                memoized["wavenumber"],
                spectrum.depth,
            )

        if "wave_reynolds_number" not in memoized:
            memoized["wave_reynolds_number"] = st4_wave_reynolds_number(
                memoized["significant_orbital_velocity"],
                memoized["significant_wave_height"] / 2,
                air,
            )

        if "mutual_angle" not in memoized:
            memoized["mutual_angle"] = (
                ((spectrum.direction - direction + 180.0) % 360.0 - 180.0) * pi / 180
            )

        critical_wave_reynolds_number = st4_crictical_reynolds_number(
            self.swell_dissipation_coefficients, memoized["significant_wave_height"]
        )

        swell_dissipation = st4_swell_dissipation(
            speed,
            memoized["mutual_angle"],
            spectrum.variance_density,
            roughness_length,
            memoized["significant_wave_height"] / 2,
            memoized["wave_reynolds_number"],
            critical_wave_reynolds_number,
            memoized["wavenumber"],
            spectrum.radian_frequency,
            memoized["significant_orbital_velocity"],
            self.swell_dissipation_coefficients,
            self.gravitational_acceleration,
            air,
            water,
        )
        return swell_dissipation


def st4_wave_reynolds_number(
    significant_orbital_velocity, significant_amplitude, air: FluidProperties = AIR
):
    return (
        4
        * significant_orbital_velocity
        * significant_amplitude
        / air.kinematic_viscosity
    )


def st4_significant_orbital_velocity(
    variance_density, radian_frequency, wavenumber, depth
):
    return 2 * sqrt(
        integrate_spectral_data(
            variance_density * radian_frequency**2 * tanh(wavenumber * depth),
            dims=SPECTRAL_DIMS,
        )
    )


def st4_crictical_reynolds_number(
    swell_dissipation_coefficients, significant_wave_height
):
    return (
        swell_dissipation_coefficients["dimensional_critical_reynolds_number"]
        / significant_wave_height
    )


def st4_swell_dissipation(
    speed,
    mutual_angle,
    variance_density,
    roughness,
    significant_amplitude,
    wave_reynolds_number,
    critical_reynolds_number,
    wavenumber,
    angular_frequency,
    significant_orbital_velocity,
    swell_dissipation_coefficients,
    gravitational_acceleration,
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
):
    ratio = air.density / water.density

    criterion = wave_reynolds_number <= critical_reynolds_number
    laminar_coeficient = swell_dissipation_coefficients["laminar_coeficient_cds"]
    laminar = -(
        laminar_coeficient
        * ratio
        * 2
        * wavenumber
        * sqrt(2 * angular_frequency * air.kinematic_viscosity)
    )

    swell_dissipation_factor = st4_swell_dissipation_factor(
        speed,
        significant_orbital_velocity,
        roughness,
        significant_amplitude,
        mutual_angle,
        swell_dissipation_coefficients,
        water,
    )

    turbulent = -(
        ratio
        * 16
        * swell_dissipation_factor
        * angular_frequency**2
        * significant_orbital_velocity
        / gravitational_acceleration
    )

    return where(criterion, laminar, turbulent) * variance_density


def st4_swell_dissipation_factor(
    speed,
    significant_orbital_velocity,
    roughness,
    significant_amplitude,
    mutual_angle,
    swell_dissipation_coefficients,
    water: FluidProperties,
):
    dissipation_factor_grant_maddsen = st4_dissipation_factor_grant_maddsen(
        roughness, significant_amplitude, swell_dissipation_coefficients, water
    )

    return swell_dissipation_coefficients["s1"] * (
        dissipation_factor_grant_maddsen
        + (
            swell_dissipation_coefficients["s3"]
            + swell_dissipation_coefficients["s2"] * cos(mutual_angle)
        )
        * speed
        / significant_orbital_velocity
    )


def st4_dissipation_factor_grant_maddsen(
    roughness,
    significant_amplitude,
    swell_dissipation_coefficients,
    water: FluidProperties = WATER,
):
    dimensionless_roughness = (
        swell_dissipation_coefficients["rz0"] * roughness / significant_amplitude
    )

    kel = kelvin(2 * sqrt(dimensionless_roughness))[1]
    return water.vonkarman_constant**2 / (abs(kel) ** 2 * 2)


def st4_effective_wave_age(
    wind_forcing_parameter,
    vonkarman_constant,
    wavenumber,
    roughness_length,
    wave_age_tuning_parameter,
):
    factor = wind_forcing_parameter + wave_age_tuning_parameter
    data = where(
        factor > 0,
        log(wavenumber * roughness_length) + vonkarman_constant / (factor),
        0,
    )
    return where(data <= 0, data, 0)


def st4_temporal_growth_rate_wave_energy(
    friction_velocity,
    wind_direction_degrees,
    wave_direction,
    vonkarman_constant,
    wavenumber,
    wavespeed,
    roughness_length,
    wave_age_tuning_parameter,
    growth_parameter_betamax,
):
    wind_forcing_parameter = st4_wind_forcing_parameter(
        wavespeed, wave_direction, friction_velocity, wind_direction_degrees
    )
    effective_wave_age = st4_effective_wave_age(
        wind_forcing_parameter,
        vonkarman_constant,
        wavenumber,
        roughness_length,
        wave_age_tuning_parameter,
    )

    return (
        growth_parameter_betamax
        / vonkarman_constant**2
        * exp(effective_wave_age)
        * effective_wave_age**4
        * wind_forcing_parameter**2
    )


def st4_wind_forcing_parameter(
    wave_speed, direction, friction_velocity, wind_direction_degrees
) -> DataArray:
    """ """
    delta = ((direction - wind_direction_degrees + 180.0) % 360.0 - 180.0) * pi / 180
    W = friction_velocity / wave_speed * cos(delta)
    return where(W > 0, W, 0)


def st4_wind_source_term_ustar(
    variance_density,
    radian_frequency,
    friction_velocity,
    wind_direction_degrees,
    wave_direction,
    wavenumber,
    wavespeed,
    roughness_length,
    wave_age_tuning_parameter,
    growth_parameter_betamax,
    water: FluidProperties = WATER,
    air: FluidProperties = AIR,
) -> DataArray:
    temporal_growth_rate_wave_energy = st4_temporal_growth_rate_wave_energy(
        friction_velocity,
        wind_direction_degrees,
        wave_direction,
        air.vonkarman_constant,
        wavenumber,
        wavespeed,
        roughness_length,
        wave_age_tuning_parameter,
        growth_parameter_betamax,
    )

    # Sin growth term
    wind_input = (
        air.density
        / water.density
        * (variance_density * radian_frequency * temporal_growth_rate_wave_energy)
    )

    return wind_input


# def gamma():
