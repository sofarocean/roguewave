import numpy
from roguewave.wavespectra.estimators.mem2 import mem2
from roguewave.wavespectra.estimators.mem import mem
from numba_progress import ProgressBar
from typing import Literal

Estimators = Literal["mem", "mem2"]


# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
def estimate_directional_spectrum_from_moments(
    e: numpy.ndarray,
    a1: numpy.ndarray,
    b1: numpy.ndarray,
    a2: numpy.ndarray,
    b2: numpy.ndarray,
    direction: numpy.ndarray,
    method: Estimators = "mem2",
    **kwargs,
) -> numpy.ndarray:
    """
    Construct a 2D directional distribution based on the directional moments and a spectral
    reconstruction method.

    :param number_of_directions: length of the directional vector for the
    2D spectrum. Directions returned are in degrees

    :param method: Choose a method in ['mem','mem2']
        mem: maximum entrophy (in the Boltzmann sense) method
        Lygre, A., & Krogstad, H. E. (1986). Explicit expression and
        fast but tends to create narrow spectra anderroneous secondary peaks.

        mem2: use entrophy (in the Shannon sense) to maximize. Likely
        best method see- Benoit, M. (1993).

    REFERENCES:
    Benoit, M. (1993). Practical comparative performance survey of methods
        used for estimating directional wave spectra from heave-pitch-roll data.
        In Coastal Engineering 1992 (pp. 62-75).

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
        directional distribution in ocean wave spectra.
        Journal of Physical Oceanography, 16(12), 2052-2060.

    """
    return (
        estimate_directional_distribution(a1, b1, a2, b2, direction, method, **kwargs)
        * e[..., None]
    )


def estimate_directional_distribution(
    a1: numpy.ndarray,
    b1: numpy.ndarray,
    a2: numpy.ndarray,
    b2: numpy.ndarray,
    direction: numpy.ndarray,
    method: Estimators = "mem2",
    **kwargs,
) -> numpy.ndarray:
    """
    Construct a 2D directional distribution based on the directional moments and a spectral
    reconstruction method.

    :param number_of_directions: length of the directional vector for the
    2D spectrum. Directions returned are in degrees

    :param method: Choose a method in ['mem','mem2']
        mem: maximum entrophy (in the Boltzmann sense) method
        Lygre, A., & Krogstad, H. E. (1986). Explicit expression and
        fast but tends to create narrow spectra anderroneous secondary peaks.

        mem2: use entrophy (in the Shannon sense) to maximize. Likely
        best method see- Benoit, M. (1993).

    REFERENCES:
    Benoit, M. (1993). Practical comparative performance survey of methods
        used for estimating directional wave spectra from heave-pitch-roll data.
        In Coastal Engineering 1992 (pp. 62-75).

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
        directional distribution in ocean wave spectra.
        Journal of Physical Oceanography, 16(12), 2052-2060.

    """

    # Jacobian to transform distribution as function of radian angles into
    # degrees.
    Jacobian = numpy.pi / 180
    direction_radians = direction * Jacobian

    if method.lower() in ["maximum_entropy_method", "mem"]:
        # reconstruct the directional distribution using the maximum entropy
        # method.
        function = mem
    elif method.lower() in ["maximum_entrophy_method2", "mem2"]:
        function = mem2
    else:
        raise Exception(f"unsupported spectral estimator method: {method}")

    output_shape = list(a1.shape) + [len(direction)]
    if a1.ndim == 1:
        input_shape = [1, a1.shape[-1]]
    else:
        input_shape = [numpy.prod(a1.shape[0:-1]), a1.shape[-1]]

    a1 = a1.reshape(input_shape)
    b1 = b1.reshape(input_shape)
    a2 = a2.reshape(input_shape)
    b2 = b2.reshape(input_shape)

    number_of_iterations = a1.shape[0]
    if number_of_iterations < 10:
        disable = True
    else:
        disable = False

    if method != "mem2":
        msg = f"Reconstructing 2d spectrum with {method} using implementation: "
    else:
        solution_method = kwargs.get("solution_method", "scipy")
        msg = f"Reconstructing 2d spectrum with {method} using solution_method {solution_method}"

    with ProgressBar(total=number_of_iterations, disable=disable, desc=msg) as progress:
        res = function(direction_radians, a1, b1, a2, b2, progress, **kwargs)

    return res.reshape(output_shape) * Jacobian
