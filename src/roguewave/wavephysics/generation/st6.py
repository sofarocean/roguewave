from roguewave.wavephysics.fluidproperties import AIR, WATER, FluidProperties
from roguewave import FrequencyDirectionSpectrum
from roguewave.wavephysics.generation import WindGeneration
from numpy import tanh, cos, pi, sqrt, log
from xarray import DataArray, where


class ST6(WindGeneration):
    def memoize(self, spectrum: FrequencyDirectionSpectrum, memoize):
        if (spectrum is not self._object) or not memoize:
            self._object = spectrum
            self.wave_speed = spectrum.wave_speed()
            self.peak_wave_speed = spectrum.peak_wave_speed()
            self.peak_angular_frequency = spectrum.peak_angular_frequency()
            self.saturation_spectrum = (
                spectrum.wavenumber**3 * spectrum.group_velocity * spectrum.e / 2 / pi
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

        memoized = memoized if memoized is not None else {}
        if "wave_speed" not in memoized:
            memoized["wave_speed"] = spectrum.wave_speed()

        if "peak_wave_speed" not in memoized:
            memoized["peak_wave_speed"] = spectrum.peak_wave_speed()

        if "peak_angular_frequency" not in memoized:
            memoized["peak_angular_frequency"] = spectrum.peak_angular_frequency()

        if "saturation_spectrum" not in memoized:
            memoized["saturation_spectrum"] = (
                spectrum.wavenumber**3 * spectrum.group_velocity * spectrum.e / 2 / pi
            )

        return st6_wind_source_term(
            spectrum.variance_density,
            speed,
            direction,
            memoized["wave_speed"],
            spectrum.direction,
            memoized["peak_wave_speed"],
            memoized["peak_angular_frequency"],
            spectrum.radian_frequency,
            memoized["saturation_spectrum"],
            air,
            water,
        )

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

        u10 = speed / air.vonkarman_constant * log(10 / roughness_length)
        return self.rate_U10(
            u10, direction, spectrum, roughness_length, air, water, memoized
        )


def st6_wind_source_term(
    spectrum,
    u10,
    wind_direction,
    wave_speed,
    direction,
    peak_wave_speed,
    peak_angular_frequency,
    radian_frequency,
    saturation_spectrum,
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
):
    """
    :param spectrum:
    :param u10:
    :param wind_direction:
    :param air:
    :param water:
    :return:
    """

    # Growth rate factor
    gamma = st6_temporal_growth_rate_wave_energy(
        u10,
        wind_direction,
        wave_speed,
        direction,
        peak_wave_speed,
        peak_angular_frequency,
        radian_frequency,
        saturation_spectrum,
    )

    # Sin growth term
    wind_input = air.density / water.density * (spectrum * radian_frequency * gamma)

    return wind_input


def st6_temporal_growth_rate_wave_energy(
    u10: DataArray,
    wind_direction: DataArray,
    wave_speed: DataArray,
    direction: DataArray,
    peak_wave_speed: DataArray,
    peak_angular_frequency: DataArray,
    radian_frequency: DataArray,
    saturation_spectrum: DataArray,
) -> DataArray:
    """
    :param u10:
    :param wind_direction:
    :param wave_speed:
    :param direction:
    :param peak_wave_speed:
    :param peak_angular_frequency:
    :param radian_frequency:
    :param saturation_spectrum:
    :param output:
    :return:
    """

    W_squared = (
        st6_wind_forcing_parameter(wave_speed, direction, u10, wind_direction) ** 2
    )
    sqrt_spectral_saturation = sqrt(
        st6_spectral_saturation(
            u10,
            peak_wave_speed,
            peak_angular_frequency,
            radian_frequency,
            saturation_spectrum,
        )
    )
    W_times_sqrt_spectral_saturation = W_squared * sqrt_spectral_saturation
    return (
        st6_sheltering_coefficient(W_times_sqrt_spectral_saturation)
        * W_times_sqrt_spectral_saturation
    )


def st6_directional_spreading_function(
    u10: DataArray, peak_wave_speed, peak_angular_frequency, radian_frequency
) -> DataArray:
    """
    Babanin & Soloviev, 1998.

    :param u10:
    :param spectrum:
    :return:
    """
    return 1.12 * (u10 / peak_wave_speed) ** (-0.5) * (
        radian_frequency / peak_angular_frequency
    ) ** -(0.95) + 1 / (2 * pi)


def st6_spectral_saturation(
    u10, peak_wave_speed, peak_angular_frequency, radian_frequency, saturation_spectrum
) -> DataArray:
    """

    :param u10:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return saturation_spectrum * st6_directional_spreading_function(
        u10, peak_wave_speed, peak_angular_frequency, radian_frequency
    )


def st6_wind_forcing_parameter(
    wave_speed, direction, u10, wind_direction_degrees
) -> DataArray:
    """

    :param spectrum:
    :param u10:
    :param wind_direction_degrees:
    :return:
    """
    delta = (direction - wind_direction_degrees + 180.0) % 360.0 - 180.0
    W = u10 * cos(delta * pi / 180) / wave_speed - 1
    return where(W > 0, W, 0)


def st6_sheltering_coefficient(
    W_times_sqrt_spectral_saturation,
) -> DataArray:
    """
    :param u10:
    :param wind_direction:
    :param spectrum:
    :return:
    """
    return 2.8 - (1 + tanh(10 * W_times_sqrt_spectral_saturation - 11))
