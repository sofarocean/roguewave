from roguewave.wavephysics.loglaw import friction_velocity_from_windspeed, loglaw
from roguewave.wavephysics.roughnesslength import create_roughness_length_estimator
from tests.test_data.spectra import get_1d_spec
from numpy.testing import assert_allclose
from xarray import ones_like


def test_ustar_from_windspeed():
    spectrum = get_1d_spec()
    U10 = ones_like(spectrum.depth) * 10

    # # Constant Charnock
    z0 = create_roughness_length_estimator("charnock_constant")
    ustar = friction_velocity_from_windspeed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0))

    # # Voermans15 Charnock
    z0 = create_roughness_length_estimator("charnock_voermans15")
    ustar = friction_velocity_from_windspeed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0))

    # Voermans16 Charnock
    z0 = create_roughness_length_estimator("charnock_voermans16")
    ustar = friction_velocity_from_windspeed(U10, spectrum, z0, elevation=10)
    assert_allclose(U10, loglaw(ustar, 10, spectrum, z0))


if __name__ == "__main__":
    test_ustar_from_windspeed()
