from roguewave.wavespectra import timeseries, parametric
import numpy
from numpy.testing import assert_allclose


def test_timeseries_spec_1d():
    sampling_frequency = 2.5
    significant_waveheight = 5

    frequencies = numpy.linspace(0, sampling_frequency, 256, endpoint=False)
    directions = numpy.linspace(0, 360, 36, endpoint=False)
    spectrum = parametric.create_parametric_spectrum(
        frequencies,
        "pm",
        0.1,
        significant_waveheight,
        directions,
        "raised_cosine",
        0,
        30,
    )
    spec1d = spectrum.as_frequency_spectrum()

    time, z = timeseries.surface_timeseries(
        "z", sampling_frequency, 400 * sampling_frequency, spec1d
    )
    time, w = timeseries.surface_timeseries(
        "w", sampling_frequency, 400 * sampling_frequency, spec1d
    )

    assert numpy.allclose(
        4 * numpy.sqrt(numpy.var(z)), significant_waveheight, rtol=1e-2, atol=1e-2
    )


def test_timeseries_spec_2d():
    sampling_frequency = 2.5
    significant_waveheight = 5

    frequencies = numpy.linspace(0, sampling_frequency, 256, endpoint=False)
    directions = numpy.linspace(0, 360, 144, endpoint=False)

    def get_sigs(direction, width):
        spectrum = parametric.create_parametric_spectrum(
            frequencies,
            "pm",
            0.1,
            significant_waveheight,
            directions,
            "raised_cosine",
            direction,
            width,
        )

        # Note due to the "standing wave effect" we need a long time series to get a proper signal.
        time, x = timeseries.surface_timeseries(
            "x", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        time, y = timeseries.surface_timeseries(
            "y", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        time, z = timeseries.surface_timeseries(
            "z", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        time, w = timeseries.surface_timeseries(
            "w", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        time, u = timeseries.surface_timeseries(
            "u", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        time, v = timeseries.surface_timeseries(
            "v", sampling_frequency, 100000 * sampling_frequency, spectrum
        )
        sigs = [u, v, w, x, y, z]
        vars = [numpy.var(f) for f in sigs]
        return sigs, vars

    # unidirectional to the east
    sigs, vars = get_sigs(0, 5)
    assert_allclose(vars[0], vars[2], rtol=0.1, atol=0.1)
    assert_allclose(vars[3], vars[5], rtol=0.1, atol=0.1)
    assert vars[1] < 5e-2, vars[1]
    assert vars[4] < 5e-2, vars[4]
    assert_allclose(4 * numpy.sqrt(vars[5]), significant_waveheight, rtol=0.1, atol=0.5)

    # unidirectional to the north
    sigs, vars = get_sigs(90, 5)
    assert_allclose(vars[1], vars[2], rtol=0.1, atol=0.1)
    assert_allclose(vars[4], vars[5], rtol=0.1, atol=0.1)
    assert vars[0] < 5e-2, vars[1]
    assert vars[3] < 5e-2, vars[3]
    assert_allclose(4 * numpy.sqrt(vars[5]), significant_waveheight, rtol=0.1, atol=0.5)


if __name__ == "__main__":
    test_timeseries_spec_1d()
    test_timeseries_spec_2d()
