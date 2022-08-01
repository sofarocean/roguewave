# Import
# =============================================================================
from datetime import datetime, timedelta

from .modelinformation import _get_resource_specification, \
    _generate_aws_key
from .timebase import timebase_forecast, \
    timebase_lead, timebase_evaluation
from typing import List, Tuple


# Main Public Functions
# =============================================================================
def generate_forecast_keys(
        variable: str,
        init_time: datetime,
        duration: timedelta,
        model_name: str) -> List[str]:
    """
    Get the AWS keys associated with a forecast for a given
    model.

    :param variable: name of the variable of interest
    :param init_time: init time of the forecast of interest
    :param duration: maximum lead time of interest
    :param model_name: model name
    :return: List of valid aws_keys.
    """
    aws_key_layout = _get_resource_specification(model_name)
    time_vector = timebase_forecast(
        init_time=init_time,
        duration=duration,
        time_configuration=aws_key_layout.model_time_configuration
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# ------------------------------------------------------------------------------


def generate_analysis_keys(
        variable:str,
        start_time: datetime,
        end_time: datetime,
        model_name: str,
) -> List[str]:
    """
    Get the AWS keys associated with the analysis results for
    a given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param model_name: model name
    :return: List of the aws_keys
    """
    aws_key_layout = _get_resource_specification(model_name)
    time_vector = timebase_lead(
        start_time,
        end_time,
        timedelta(hours=0),
        time_configuration=aws_key_layout.model_time_configuration,
        exact=False
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# -----------------------------------------------------------------------------


def generate_lead_keys(
        variable: str,
        start_time: datetime, end_time: datetime,
        lead_time: timedelta, model_name: str,
        exact=False) -> List[str]:
    """
    Get the AWS keys associated with a given lead time for a
    given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param lead_time: Lead time of interest
    :param model_name: model name
    :param exact: If true, generate keys exactly at given lead time. Otherwise
    generate keys as close as possible.
    :return: List of the aws_keys.
    """
    aws_key_layout = _get_resource_specification(model_name)
    time_vector = timebase_lead(
        start_time=start_time,
        end_time=end_time,
        lead_time=lead_time,
        time_configuration=aws_key_layout.model_time_configuration,
        exact=exact
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# -----------------------------------------------------------------------------


def generate_evaluation_time_keys(
        variable: str,
        evaluation_time: datetime,
        model_name: str,
        maximum_lead_time: timedelta = None
) -> List[str]:
    """
    Get the AWS keys associated with a given evaluation time for a
    given model.

    :param variable: name of the variable of interest
    :param evaluation_time: evaluation time of interest
    :param model_name: model name
    :param maximum_lead_time: maximum lead time of interest
    :return: list of aws keys
    """
    aws_key_layout = _get_resource_specification(model_name)
    if maximum_lead_time is None:
        maximum_lead_time = aws_key_layout.model_time_configuration.duration

    time_vector = timebase_evaluation(
        evaluation_time=evaluation_time,
        maximum_lead_time=maximum_lead_time,
        time_configuration=aws_key_layout.model_time_configuration
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# -----------------------------------------------------------------------------


# Internal module functions
# =============================================================================
def _generate_keys(time_vector, variable, aws_key_layout) -> List[str]:
    aws_keys = []
    for time in time_vector:
        aws_keys.append(
            _generate_aws_key(
                variable=variable,
                init_time=time[0],
                forecast_hour=time[1],
                model=aws_key_layout)
        )
    return aws_keys

