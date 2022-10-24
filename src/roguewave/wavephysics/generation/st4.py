from roguewave.wavephysics.fluidproperties import (
    AIR,
    WATER,
    FluidProperties,
    GRAVITATIONAL_ACCELERATION,
)
from roguewave import FrequencyDirectionSpectrum
from roguewave.wavephysics.generation import WindGeneration
from numpy import cos, pi, log, exp
from xarray import DataArray, where


class ST4(WindGeneration):
    def __init__(
        self,
        wave_age_tuning_parameter=0.006,
        growth_parameter_betamax=0.5,
        gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
        **kwargs,
    ):
        super(ST4, self).__init__(**kwargs)
        self.wave_age_tuning_parameter = wave_age_tuning_parameter
        self.growth_parameter_betamax = growth_parameter_betamax
        self.gravitational_acceleration = gravitational_acceleration

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
        return wind_input

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


def st4_effective_wave_age(
    wind_forcing_parameter,
    vonkarman_constant,
    wavenumber,
    roughness_length,
    wave_age_tuning_parameter,
):
    factor = wind_forcing_parameter
    data = where(
        factor > 0,
        log(wavenumber * roughness_length)
        + vonkarman_constant / (factor + wave_age_tuning_parameter),
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
