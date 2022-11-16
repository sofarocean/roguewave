from numba import njit
from numpy import empty_like
from numpy.typing import NDArray


@njit(cache=True)
def integrate(time: NDArray, signal: NDArray) -> NDArray:
    coef = [3 / 8, 19 / 24, -5 / 24, 1 / 24]

    integrated_signal = empty_like(signal)
    integrated_signal[0] = 0

    # Start with Trapezoidal rule
    for ii in range(1, 3):
        dt = time[ii] - time[ii - 1]
        integrated_signal[ii] = (
            integrated_signal[ii - 1] + (signal[ii] + signal[ii - 1]) / 2 * dt
        )

    # Then apply Adams Moulton
    for ii in range(3, len(signal)):
        dt = time[ii] - time[ii - 1]
        integrated_signal[ii] = (
            integrated_signal[ii - 1]
            + coef[0] * signal[ii] * dt
            + coef[1] * signal[ii - 1] * dt
            + coef[2] * signal[ii - 2] * dt
            + coef[3] * signal[ii - 3] * dt
        )
    return integrated_signal
