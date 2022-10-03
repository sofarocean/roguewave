from roguewave.wavespectra.parametric import (
    create_frequency_shape,
    create_directional_shape,
    create_parametric_spectrum,
)
from numpy.testing import assert_allclose
from numpy import linspace, argmax, sum, trapz


def test_raised_cosine():
    distribution = create_directional_shape(
        "raised_cosine", mean_direction_degrees=20, width_degrees=28.64
    )
    assert_allclose(distribution._power, 2, rtol=0.002, atol=0.003)

    angles = linspace(0, 360, 36, endpoint=False)
    D = distribution.values(angles)
    assert argmax(D) == 2, f"maximum at {angles[argmax((D))]}, not at 20"
    assert_allclose(sum(D) * 10, 1, rtol=0.001, atol=0.001)


def test_pierson_moskowitz():
    distribution = create_frequency_shape("pm", peak_frequency_hertz=0.1, m0=1)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    E = distribution.values(frequency)

    # Test that the peak is at the right location
    assert argmax(E) == 9, f"maximum at {frequency[argmax((E))]}, not at 0.1"

    # Test that it integrates to 1
    assert_allclose(trapz(E, frequency), 1, rtol=0.001, atol=0.001)


def test_create_parametric_spectrum():
    angles = linspace(0, 360, 36, endpoint=False)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    spectrum = create_parametric_spectrum(
        frequency,
        "pm",
        peak_frequency_hertz=0.1,
        significant_wave_height=2,
        direction_degrees=angles,
        direction_shape="raised_cosine",
        mean_direction_degrees=20,
        width_degrees=10,
    )

    assert_allclose(spectrum.significant_waveheight.values, 2, rtol=0.001, atol=0.001)
    assert_allclose(spectrum.peak_frequency().values, 0.1, rtol=0.001, atol=0.001)
    assert_allclose(spectrum.mean_direction().values, 20, rtol=0.001, atol=0.001)

    # To note- the definition of spectral width is slightly different for the theoretical value used in the
    # shape. Hence the difference.
    assert_allclose(
        spectrum.mean_directional_spread().values, 10.116, rtol=0.001, atol=0.001
    )


if __name__ == "__main__":
    test_raised_cosine()
    test_pierson_moskowitz()
    test_create_parametric_spectrum()
