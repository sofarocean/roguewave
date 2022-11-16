from roguewave.timeseries_analysis.filtering import (
    sos_filter,
    spike_filter,
    exponential_filter,
)
from roguewave.timeseries_analysis.time_integration import integrate
from numpy.typing import NDArray
from typing import Union, List, Callable
from scipy.signal import butter

sos = butter(4, 0.04, btype="high", output="sos", fs=2.5)


def pipeline(time: NDArray, signal: NDArray, stages=None) -> NDArray:
    if stages is None:
        stages = [
            ("spike_filter", None),
            ("exponential_filter", None),
            ("integrate", None),
            ("sos_filtfilt", None),
        ]

    stages = create_pipeline(stages)
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
    elif stage == "spike_filter":
        return lambda time, signal: spike_filter(time, signal)
    elif stage == "exponential_filter":
        return lambda time, signal: exponential_filter(time, signal)
    else:
        raise Exception("unknown filter")


def create_pipeline(stages: List[Union[str, STAGE_SIGNATURE]]) -> List[STAGE_SIGNATURE]:
    functions = []
    for stage in stages:
        if isinstance(stage, tuple):
            functions.append(create_stage(*stage))
        else:
            functions.append(stage)
    return functions
