from roguewave.timeseries_analysis.filtering import (
    sos_filter,
    spike_filter,
    exponential_delta_filter,
    exponential_filter,
    cumulative_filter,
)
from roguewave import FrequencySpectrum
from roguewave.timeseries_analysis.time_integration import (
    integrate,
    complex_response,
    DEFAULT_N,
    DEFAULT_ORDER,
    cumulative_distance,
)
from numpy.typing import NDArray
from numpy import issubdtype, datetime64, ones
from roguewave.tools.time import datetime64_to_timestamp
from scipy.signal import butter
from numba import types
from numba.typed import Dict
from numbers import Number
from .welch import estimate_frequency_spectrum


DEFAULT_VELOCITY_PIPELINE = [
    ("spike", {"rejection_threshold": 9.81}),
    (
        "exponential_delta",
        {"smoothing_factor": 0.004, "maximum_gap_size": 3, "sampling_frequency": 2.5},
    ),
    ("integrate", {"order": DEFAULT_ORDER, "n": DEFAULT_N, "start_value": 0.0}),
    (
        "sos",
        {
            "direction": "filtfilt",
            "sos": butter(4, 0.04, btype="high", output="sos", fs=2.5),
        },
    ),
]

DEFAULT_DISPLACEMENT_PIPELINE = [
    (
        "exponential_delta",
        {"smoothing_factor": 0.004, "maximum_gap_size": 3, "sampling_frequency": 2.5},
    ),
    (
        "sos_filtfilt",
        {
            "direction": "filtfilt",
            "sos": butter(4, 0.04, btype="high", output="sos", fs=2.5),
        },
    ),
]

DEFAULT_SPOTTER_PIPELINE = DEFAULT_VELOCITY_PIPELINE


def pipeline(time: NDArray, signal: NDArray, stages=None) -> NDArray:
    if stages is None:
        stages = DEFAULT_SPOTTER_PIPELINE

    for index in range(0, len(stages)):
        value = stages[index]
        if isinstance(value, str):
            stages[index] = (value, {})

        elif isinstance(value, tuple):
            if value[1] is None:
                stages[index] = (value[0], {})

    if issubdtype(time.dtype, datetime64):
        time = datetime64_to_timestamp(time)

    for index, stage in enumerate(stages):
        signal = apply_filter(stage[0], time, signal, **stage[1])

    return signal


def spectral_pipeline(
    epoch_time: NDArray,
    x_or_u: NDArray,
    y_or_v: NDArray,
    z_or_w: NDArray,
    x_or_u_pipeline=None,
    y_or_v_pipeline=None,
    z_or_w_pipeline=None,
    latlon_input=False,
    window=None,
    segment_length_seconds=1800,
    sampling_frequency=2.5,
    spectral_window=None,
    response_correction=False,
) -> FrequencySpectrum:

    """
    Calculate the 1d frequency wave-spectrum including directional moments a1,b1,a2,b2 based on
    the raw input signals. The signals are processed according to the given pipeline prior to
    calculation of the co-spectra. If no pipelines are provided we use the default pipelines on
    Spotter which take as input: horizontal displacements x,y and vertical velocity w to produce
    the spectrum.

    :param time:
    :param x_or_u:
    :param y_or_v:
    :param z_or_w:
    :param x_or_u_pipeline:
    :param y_or_v_pipeline:
    :param z_or_w_pipeline:
    :param latlon_input:
    :param window:
    :param segment_length_seconds:
    :param spectral_window:
    :param response_correction:
    :return:
    """

    if latlon_input:
        x_or_u, y_or_v = cumulative_distance(x_or_u, y_or_v)

    if issubdtype(epoch_time.dtype, datetime64):
        epoch_time = datetime64_to_timestamp(epoch_time)

    if x_or_u_pipeline is None:
        x_or_u_pipeline = DEFAULT_DISPLACEMENT_PIPELINE
    if y_or_v_pipeline is None:
        y_or_v_pipeline = DEFAULT_DISPLACEMENT_PIPELINE
    if z_or_w_pipeline is None:
        z_or_w_pipeline = DEFAULT_VELOCITY_PIPELINE

    x = pipeline(epoch_time, x_or_u, x_or_u_pipeline)
    y = pipeline(epoch_time, y_or_v, y_or_v_pipeline)
    z = pipeline(epoch_time, z_or_w, z_or_w_pipeline)

    # Integration correction response funcitons
    response_functions = (
        get_response_correction(x_or_u_pipeline, apply=response_correction),
        get_response_correction(y_or_v_pipeline, apply=response_correction),
        get_response_correction(z_or_w_pipeline, apply=response_correction),
    )

    return estimate_frequency_spectrum(
        epoch_time,
        x,
        y,
        z,
        window,
        segment_length_seconds,
        sampling_frequency=sampling_frequency,
        spectral_window=spectral_window,
        response_functions=response_functions,
    )


def apply_filter(
    name: str, time_seconds: NDArray, signal: NDArray, **kwargs
) -> NDArray:
    """
    Apply the named filter to the signal and return the result.

    :param name: name of the filter to apply
    :param time_seconds: NDArray of time in seconds
    :param signal: NDArray of the signal (same length as time signal)
    :param kwargs: keyword filter options.
    :return: filtered signal
    """

    if name == "sos":
        if "direction" not in kwargs:
            kwargs["direction"] = "forward"
        return sos_filter(signal, **kwargs)

    elif name == "integrate":
        return integrate(
            time_seconds,
            signal,
            kwargs.get("order", DEFAULT_ORDER),
            kwargs.get("n", DEFAULT_N),
            kwargs.get("start_value", 0.0),
        )

    elif name == "spike":
        options = to_numba_kwargs(kwargs)
        return spike_filter(time_seconds, signal, options=options)

    elif name == "exponential_delta":
        options = to_numba_kwargs(kwargs)
        return exponential_delta_filter(time_seconds, signal, options=options)

    elif name == "cumulative":
        options = to_numba_kwargs(kwargs)
        return cumulative_filter(signal, options)

    elif name == "exponential":
        options = to_numba_kwargs(kwargs)
        return exponential_filter(time_seconds, signal, options)

    elif name == "identity":
        return signal

    else:
        raise Exception(f"unknown filter: {name}")


def to_numba_kwargs(kwargs):
    options = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for k, v in kwargs.items():
        if isinstance(v, Number):
            options[k] = v

    return options


def get_response_correction(pipeline, apply=True):
    if not apply:
        return lambda f: ones(len(f), dtype="complex_")

    for stage in pipeline:
        if stage[0] == "integrate":
            return lambda f: 1 / complex_response(
                f, stage[1].get("order", DEFAULT_ORDER), stage[1].get("n", DEFAULT_N)
            )
    else:
        return lambda f: ones(len(f), dtype="complex_")
