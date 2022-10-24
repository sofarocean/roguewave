from roguewave.wavephysics.loglaw import friction_velocity_from_speed, loglaw
from roguewave.wavephysics.momentumflux import create_roughness_length_estimator
from roguewave.wavephysics.fluidproperties import AIR
from roguewave.log import set_log_to_console
from tests.test_data.spectra import get_1d_spec
from numpy.testing import assert_allclose
from xarray import ones_like


def test_ustar_from_windspeed():
    spectrum = get_1d_spec()
    U10 = ones_like(spectrum.depth) * 10

    # Constant Charnock
    z0 = create_roughness_length_estimator("charnock_constant")
    ustar = friction_velocity_from_speed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0, AIR))

    # Voermans15 Charnock
    z0 = create_roughness_length_estimator("charnock_voermans15")
    ustar = friction_velocity_from_speed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0, AIR))

    # Voermans16 Charnock
    z0 = create_roughness_length_estimator("charnock_voermans16")
    ustar = friction_velocity_from_speed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0, AIR))

    z0 = create_roughness_length_estimator("charnock_janssen")
    spec = spectrum.as_frequency_direction_spectrum(36)
    ustar = friction_velocity_from_speed(
        U10, spec, z0, direction_degrees=0, elevation=10
    )
    assert_allclose(
        U10, loglaw(ustar, 10, spec, z0, AIR, direction_degrees=0), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    set_log_to_console()
    test_ustar_from_windspeed()
