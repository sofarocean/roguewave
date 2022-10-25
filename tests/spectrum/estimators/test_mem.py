from roguewave.wavespectra.parametric import create_parametric_spectrum
import numpy
from roguewave import concatenate_spectra, FrequencyDirectionSpectrum
from datetime import datetime, timezone
from numpy.testing import assert_allclose


def get_2d_spec() -> FrequencyDirectionSpectrum:
    freq = numpy.linspace(0, 1, 20)
    dir = numpy.linspace(0, 360, 36, endpoint=False)
    time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    spec = create_parametric_spectrum(
        freq, "pm", 0.1, 1, dir, "raised_cosine", 35, 20, depth=numpy.inf, time=time
    )
    return concatenate_spectra([spec], "time")


def test_mem():
    moments = ["a1", "b1", "a2", "b2"]

    spec2d = get_2d_spec()
    spec1d = spec2d.as_frequency_spectrum()
    reconstructed = spec1d.as_frequency_direction_spectrum(
        36, method="mem", solution_method="scipy"
    )

    for moment in moments:
        x = getattr(spec2d, moment)
        y = getattr(reconstructed, moment)
        assert_allclose(y, x, rtol=1e-2, atol=1e-2)
    pass


if __name__ == "__main__":
    test_mem()
