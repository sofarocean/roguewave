from roguewave.timeseries_analysis.zerocrossing import maximum_wave_height, zero_crossing_period, \
    zero_crossing_wave_heights
import numpy as np


def get_signal():
    t = np.linspace(0, 100, 1000)

    f = np.array([0.1, 0.15, 0.25, 0.4])
    a = np.array([3, 2, 1, 0.5])

    sig = np.zeros_like(t)
    for amp, freq in zip(a, f):
        sig = sig + amp * np.sin(2 * np.pi * freq * t)

    return t, sig


def test_maximum_wave_height():
    t, signal = get_signal()

    max_wave_height = maximum_wave_height(signal)
    np.testing.assert_allclose(max_wave_height, 9.79, atol=1e-1, rtol=1e-1)


def test_zero_crossing_period():
    t, signal = get_signal()
    result = [5.78785186, 5.9244021, 8.30721477, 5.78838443, 5.92495274, 8.20872852, 5.88633251, 5.9254793,
              8.2076879, 5.88685908, 5.82322719, 8.30940185, 5.88740971, 5.82375975]

    _zero_crossing_period = zero_crossing_period(t, signal)
    print(_zero_crossing_period)
    np.testing.assert_allclose(_zero_crossing_period, result, atol=1e-1, rtol=1e-1)


def test_zero_crossing_wave_heights():
    t, signal = get_signal()
    result = [4.5606385, 4.55701816, 9.79219267, 4.56517232, 4.56302325, 9.78699159
        , 4.56595773, 4.56595773, 9.78699159, 4.56302325, 4.56517232, 9.79219267
        , 4.55701816, 4.5606385]
    _zero_crossing_wave_heights = zero_crossing_wave_heights(signal)

    np.testing.assert_allclose(_zero_crossing_wave_heights, result, atol=1e-1, rtol=1e-1)
