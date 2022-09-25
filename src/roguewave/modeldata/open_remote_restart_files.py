from roguewave.wavewatch3 import RestartFileTimeStack, open_restart_file_stack
from .timebase import TimeSlice
from .keygeneration import generate_uris


def open_remote_restart_file(
    variable: str,
    time_slice: TimeSlice,
    model_name: str,
    model_definition_file: str,
    cache=True,
    cache_name: str = None,
) -> RestartFileTimeStack:

    aws_keys = generate_uris(variable, time_slice, model_name)
    return open_restart_file_stack(aws_keys, model_definition_file, cache, cache_name)
