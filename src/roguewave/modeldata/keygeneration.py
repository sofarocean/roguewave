# Import
# =============================================================================
from datetime import datetime, timedelta
from .timebase import ModelTimeConfiguration, timebase_forecast, \
    timebase_lead, timebase_evaluation
import os
from typing import List, Tuple
import json

# Model private variables
# =============================================================================

# Load the model configuration file.
try:
    with open(os.path.expanduser('~/model_configuration.json'), 'rb') as file:
        MODELS = json.load(file)
except FileNotFoundError as e:
    MODELS = {}


# Classes
# =============================================================================
class _ModelAwsKeyLayout:
    """
    Class containing the key template and other necessary information to
    reconstruct keys on AWS.
    """

    def __init__(self,
                 name: str,
                 bucket: str,
                 key_template: str,
                 filetype: str = 'netcdf',
                 model_time_configuration: dict = None):

        """
        :param name: Model name
        :param bucket: bucket
        :param key_template: template of the AWS key
        :param filetype: filetype of the data
        :param model_time_configuration: Description on the timebase of the
        model (how often is the mode run, at what interval is output etc.).
        """
        self.name = name
        self.bucket = bucket
        self.key_template = key_template
        self.filetype = filetype

        if model_time_configuration is None:
            self.model_time_configuration = ModelTimeConfiguration()
        else:
            self.model_time_configuration = ModelTimeConfiguration(
                **model_time_configuration
            )


# Main Public Functions
# =============================================================================
def generate_forecast_keys_and_valid_times(
        variable, init_time: datetime,
        duration: timedelta,
        model_name: str) -> Tuple[List[str], List[datetime]]:
    """
    Get the AWS keys and valid times associated with a forecast for a given
    model.

    :param variable: name of the variable of interest
    :param init_time: init time of the forecast of interest
    :param duration: maximum lead time of interest
    :param model_name: model name
    :return: two results, where the first is a list of the aws_keys and the
    second a list of the valid times that apply to the keys.
    """
    aws_key_layout = _get_model_aws_layout(model_name)
    time_vector = timebase_forecast(
        init_time=init_time,
        duration=duration,
        time_configuration=aws_key_layout.model_time_configuration
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# ------------------------------------------------------------------------------


def generate_analysis_keys_and_valid_times(
        variable, start_time: datetime, end_time: datetime,
        model_name: str,
) -> Tuple[List[str], List[datetime]]:
    """
    Get the AWS keys and valid times associated with the analysis results for
    a given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param model_name: model name
    :return: two results, where the first is a list of the aws_keys and the
    second a list of the valid times that apply to the keys.
    """
    aws_key_layout = _get_model_aws_layout(model_name)
    time_vector = timebase_lead(
        start_time,
        end_time,
        timedelta(hours=0),
        time_configuration=aws_key_layout.model_time_configuration,
        exact=False
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# -----------------------------------------------------------------------------


def generate_lead_keys_and_valid_times(
        variable, start_time: datetime, end_time: datetime,
        lead_time: timedelta, model_name: str,
        exact=False) -> Tuple[List[str], List[datetime]]:
    """
    Get the AWS keys and valid times associated with a given lead time for a
    given model.

    :param variable: name of the variable of interest
    :param start_time: Start of time interval of interest
    :param end_time: End of time interval of interest
    :param lead_time: Lead time of interest
    :param model_name: model name
    :param exact: If true, generate keys exactly at given lead time. Otherwise
    generate keys as close as possible.
    :return: two results, where the first is a list of the aws_keys and the
    second a list of the valid times that apply to the keys.
    """
    aws_key_layout = _get_model_aws_layout(model_name)
    time_vector = timebase_lead(
        start_time=start_time,
        end_time=end_time,
        lead_time=lead_time,
        time_configuration=aws_key_layout.model_time_configuration,
        exact=exact
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# -----------------------------------------------------------------------------


def generate_evaluation_time_keys_and_valid_times(
        variable,
        evaluation_time, model_name: str,
        maximum_lead_time: timedelta = None
) -> Tuple[List[str], List[datetime]]:
    """

    :param variable: name of the variable of interest
    :param evaluation_time: evaluation time of interest
    :param model_name: model name
    :param maximum_lead_time: maximum lead time of interest
    :return:
    """
    aws_key_layout = _get_model_aws_layout(model_name)
    if maximum_lead_time is None:
        maximum_lead_time = aws_key_layout.model_time_configuration.duration

    time_vector = timebase_evaluation(
        evaluation_time=evaluation_time,
        maximum_lead_time=maximum_lead_time,
        time_configuration=aws_key_layout.model_time_configuration
    )

    return _generate_keys(time_vector, variable, aws_key_layout)


# ------------------------------------------------------------------------------


# Internal module functions
# =============================================================================
def _generate_keys(time_vector, variable, aws_key_layout):
    aws_keys = []
    valid_time = []
    for time in time_vector:
        aws_keys.append(
            _generate_aws_key(
                variable=variable,
                init_time=time[0],
                forecast_hour=time[1],
                model=aws_key_layout)
        )
        valid_time.append(time[0] + time[1])
    return aws_keys, valid_time


def _generate_aws_key(variable, init_time: datetime, forecast_hour: timedelta,
                      model: _ModelAwsKeyLayout) -> str:
    """
    Function that generates a valid AWS key for the given model that represents
    date from the model initialized at the given init_time and with the given
    forecast_hour or lead time.

    :param variable: variable name we want to retrieve (as used in key, if
        applicable, set to None otherwise)
    :param init_time: init time of the model we want to retrieve
    :param forecast_hour: the forecast hour of the model we want to retrieve
    :param model: Model() information on where data is stored.
    :return: A valid aws key for the given model.
    """

    # Helper functions. These are local for now and tied to this function
    # alone. Hence the encapsulation.
    def _replace(_string: str, key: str, value: str):
        """
        Replace a key in string with value
        :param _string: target
        :param key:  string to replace
        :param value:  string to replace with
        :return: string
        """
        return _string.replace("{" + key + "}", value)

    #

    def _time(_string, key, value: datetime):
        """
        Replace {key:fmt} with value.strftime(fmt)
        :param _string: target
        :param key:  either "init_time" or "valid_time"
        :param value: init_time or valid_time as datetime
        :return: string
        """

        # There may be multiple instances that need to be replaced:
        while key in _string:
            # Get the time format string
            _format = _find_between(_string, '{' + key + ':', '}')
            # get the text we want to replace
            to_replace = key + ':' + _format
            # what we want to replace it with (the formatted time)
            replace_with = value.strftime(_format)
            # replace
            _string = _replace(_string, to_replace, replace_with)
        return _string

    #

    def _ecmwf_special(_string, key, _init_time: datetime,
                       _forecast_hour: timedelta):
        """
        Special parsing only valid for ecmwf formatted output.
        :param _string: string
        :param key:  ecmwf_subcycle_string or ecmwf_subcycle_string
        :param _init_time: init time as datetime
        :param _forecast_hour: forecast hour as timedelta
        :return:
        """

        if key == 'ecmwf_subcycle_string':
            # the suffix of ecmwf forecasts is either `001` for all forecast
            # hours exept the zero hour.
            if _forecast_hour == timedelta(hours=0):
                ecmwf_lead_zero_indicator = '011'
            else:
                ecmwf_lead_zero_indicator = '001'
            return _replace(_string, key, ecmwf_lead_zero_indicator)

        elif key == 'ecmwf_subcycle_string':
            # the prefix of ecmwf forecasts is A2P for 10 day forecasts (00Z
            # and # 12Z cycles), or A2Q for the 4 day forecasts (06Z and 18Z
            # cycles)
            if (_init_time.hour == 0) or (_init_time.hour == 12):
                ecmwf_subcycle_string = 'A2P'
            else:
                ecmwf_subcycle_string = 'A2Q'
            return _replace(_string, key, ecmwf_subcycle_string)
        else:
            raise Exception(f'Unknown key {key}')

    # ---------------------

    #
    # Purpose:
    #
    #   The target template key has the form
    #   "t/{init_time:%Y%m%d}/{init_time:%H}/t/{forecasthour}.{variable}.nc"
    #   here everything between {...} defines a token that needs to be replaced
    #   at executing time by a value. E.g. {init_time:%Y%m%d} denotes we want
    #   to replace this with the init_time formated according to %Y%m%d. This
    #   function does the parsing/replacement.
    #
    # Implementation:
    #
    #  We define a set of properties below, where the key denotes the keyword
    #  we want to replace (init_time, forecast_hour etc.), and the value is
    #  a dictionary with a method denoting the function handling the
    #  replacement and a set of arguments in addition to the string and the key
    #  to be replaced.
    valid_time = init_time + forecast_hour

    # Define keywords
    keywords = {
        "bucket": {'method': _replace, 'args': (model.bucket,)},
        "forecasthour": {
            'method': _replace, 'args': (
                "{hours:03d}".format(
                    hours=int(forecast_hour.total_seconds() // 3600)
                ),
            )
        },
        "init_time": {'method': _time, 'args': (init_time,)},
        "valid_time": {'method': _time, 'args': (valid_time,)},
        "variable": {'method': _replace, 'args': (variable,)},
        "ecmwf_subcycle_string": {
            'method': _ecmwf_special, 'args': (init_time, forecast_hour)
        },
        "ecmwf_lead_zero_indicator": {
            'method': _ecmwf_special, 'args': (init_time, forecast_hour)
        }
    }

    # Loop over properties and invoke method to replace keywords.
    string = model.key_template
    for keyword, action in keywords.items():
        if keyword in string:
            string = action['method'](string, keyword, *action['args'])

    return f"{model.bucket}/{string}"


# Helper functions
# =============================================================================
def _find_between(s: str, first: str, last: str) -> str:
    """
     Simple function that returns the string between the "first" and "last"
     token. E.g. if first='{' and last=']' with string 'asdd{THIS]asdsd'
     this returns 'THIS'.
    :param s: string
    :param first: token signifying start
    :param last:  token signifying end
    :return:  string
    """

    # lets avoid regex for now :-)
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


# -----------------------------------------------------------------------------


def _get_model_aws_layout(name) -> _ModelAwsKeyLayout:
    """
    Return the object representing key layout on aws
    :param name:
    :return: Model
    """
    return _ModelAwsKeyLayout(name=name, **MODELS[name])

# -----------------------------------------------------------------------------
