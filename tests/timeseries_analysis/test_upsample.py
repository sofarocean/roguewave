from roguewave.timeseries_analysis.sampling import upsample
from numpy import sin, pi, linspace
from numpy.testing import assert_allclose


def test_upsample():
    low_res_time = linspace(0, 100, 250, endpoint=False)
    high_res_time = linspace(0, 100, 2500, endpoint=False)
    frequency = 2 * pi

    low_res = sin(frequency * low_res_time)
    high_res = sin(frequency * high_res_time)
    time, upsampled = upsample(low_res, 10)

    assert_allclose(upsampled, high_res, atol=1e-10, rtol=1e10)


if __name__ == "__main__":
    test_upsample()
