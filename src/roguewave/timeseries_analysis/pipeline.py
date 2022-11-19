from roguewave.timeseries_analysis.filtering import (
    sos_filter,
    spike_filter,
    exponential_delta_filter,
    exponential_filter,
    cumulative_filter,
)
from roguewave.timeseries_analysis.time_integration import integrate
from numpy.typing import NDArray
from numpy import issubdtype, datetime64
from roguewave.tools.time import datetime64_to_timestamp
from typing import Union, List, Callable
from scipy.signal import butter

sos = butter(4, 0.04, btype="high", output="sos", fs=2.5)


def pipeline(time: NDArray, signal: NDArray, stages=None) -> NDArray:
    if stages is None:
        stages = [
            ("spike", None),
            ("exponential_delta", None),
            ("integrate", None),
            ("sos_filtfilt", None),
        ]

    stages = _create_pipeline(stages)

    if issubdtype(time.dtype, datetime64):
        time = datetime64_to_timestamp(time)

    for stage in stages:
        signal = stage(time, signal)

    return signal


STAGE_SIGNATURE = Callable[[NDArray, NDArray], NDArray]


def create_stage(stage: str, stage_opt=None) -> STAGE_SIGNATURE:
    if stage == "sos_forward":
        return lambda time, signal: sos_filter(signal, "forward", stage_opt)
    elif stage == "sos_backward":
        return lambda time, signal: sos_filter(signal, "backward", stage_opt)
    elif stage == "sos_filtfilt":
        return lambda time, signal: sos_filter(signal, "filtfilt", stage_opt)
    elif stage == "integrate":
        return lambda time, signal: integrate(time, signal)
    elif stage == "spike":
        return lambda time, signal: spike_filter(time, signal)
    elif stage == "exponential_delta":
        return lambda time, signal: exponential_delta_filter(time, signal)
    elif stage == "cumulative":
        return lambda time, signal: cumulative_filter(time, signal)
    elif stage == "exponential":
        return lambda time, signal: exponential_filter(time, signal)
    else:
        raise Exception(f"unknown filter: {stage}")


def _create_pipeline(
    stages: List[Union[str, STAGE_SIGNATURE]]
) -> List[STAGE_SIGNATURE]:
    functions = []
    for stage in stages:
        if isinstance(stage, tuple):
            functions.append(create_stage(stage=stage[0], stage_opt=stage[1]))
        elif isinstance(stage, str):
            functions.append(create_stage(stage=stage, stage_opt=None))
        else:
            functions.append(stage)
    return functions
