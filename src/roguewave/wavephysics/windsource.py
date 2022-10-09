from .fluidproperties import AIR, WATER, FluidProperties, GRAVITATIONAL_ACCELERATION
from roguewave import WaveSpectrum, FrequencySpectrum, FrequencyDirectionSpectrum
from numpy import tanh, cos, pi, sqrt
from xarray import DataArray, where


def wind_source_term(
    spectrum: WaveSpectrum,
    u10,
    wind_direction=DataArray(data=0),
    air: FluidProperties = AIR,
    water: FluidProperties = WATER,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
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
    gamma = temporal_growth_rate_wave_energy(
        u10, wind_direction, spectrum, gravitational_acceleration
    )

    # Sin growth term
    wind_input = (
        air.density
        / water.density
        * (spectrum.variance_density * spectrum.radian_frequency * gamma)
    )

    if isinstance(spectrum, FrequencySpectrum):
        return wind_input

    elif isinstance(spectrum, FrequencyDirectionSpectrum):
        return (wind_input * spectrum.direction_step()).sum(dim="direction")


def temporal_growth_rate_wave_energy(
    u10: DataArray,
    wind_direction: DataArray,
    spectrum: WaveSpectrum,
    gravitational_acceleration=GRAVITATIONAL_ACCELERATION,
) -> DataArray:
    """

    :param u10:
    :param wind_direction:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    return (
        sheltering_coefficient(u10, wind_direction, spectrum)
        * sqrt(
            spectral_saturation(
                u10, spectrum, gravitational_acceleration=gravitational_acceleration
            )
        )
        * wind_forcing_parameter(spectrum, u10, wind_direction) ** 2
    )


def directional_spreading_function(u10: DataArray, spectrum: WaveSpectrum) -> DataArray:
    """
    Babanin & Soloviev, 1998.

    :param u10:
    :param spectrum:
    :return:
    """
    cp = spectrum.peak_wave_speed()
    peak_omega = spectrum.peak_angukar_frequency()
    omega = spectrum.radian_frequency
    return 1.12 * (u10 / cp) ** (-0.5) * (omega / peak_omega) ** -(0.95) + 1 / (2 * pi)


def spectral_saturation(
    u10, spectrum: WaveSpectrum, gravitational_acceleration=9.81
) -> DataArray:
    """

    :param u10:
    :param spectrum:
    :param gravitational_acceleration:
    :return:
    """
    omega = spectrum.radian_frequency
    jac_freq_to_ang_freq = 1 / 2 / pi
    if isinstance(spectrum, FrequencyDirectionSpectrum):
        E = (spectrum.variance_density * spectrum.direction_step()).sum(
            dim="direction"
        ) * jac_freq_to_ang_freq

    else:
        E = spectrum.variance_density * jac_freq_to_ang_freq

    return (
        omega**5 * E / 2 / gravitational_acceleration**2
    ) * directional_spreading_function(u10, spectrum)


def wind_forcing_parameter(
    spectrum: WaveSpectrum, u10, wind_direction_degrees
) -> DataArray:
    """

    :param spectrum:
    :param u10:
    :param wind_direction_degrees:
    :return:
    """
    if isinstance(spectrum, FrequencySpectrum):
        W = u10 / spectrum.wave_speed() - 1

    elif isinstance(spectrum, FrequencyDirectionSpectrum):
        delta = (spectrum.direction - wind_direction_degrees + 180.0) % 360.0 - 180.0
        W = u10 * cos(delta * pi / 180) / spectrum.wave_speed() - 1

    else:
        raise ValueError("unknown spectral object")

    return where(W > 0, W, 0)


def sheltering_coefficient(
    u10: DataArray, wind_direction: DataArray, spectrum: WaveSpectrum
) -> DataArray:
    """

    :param u10:
    :param wind_direction:
    :param spectrum:
    :return:
    """
    W = wind_forcing_parameter(spectrum, u10, wind_direction)
    return 2.8 - (1 + tanh(10 * sqrt(spectral_saturation(u10, spectrum)) * W**2 - 11))
