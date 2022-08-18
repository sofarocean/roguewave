# Import
# =============================================================================
from datetime import datetime, timedelta

from .modelinformation import _get_resource_specification, \
    _RemoteResourceSpecification, _find_between
from .timebase import TimeSlice
from typing import List


# Main Public Functions
# =============================================================================
def generate_uris(
        variable: str,
        time_slice: TimeSlice,
        model_name: str) -> List[str]:
    """
    Get the AWS keys associated with a time slice for a given
    model.

    :param variable: name of the variable of interest
    :param time_slice: how we slice the forecast time
    :param model_name: model name
    :return: List of valid uris.
    """
    resource = _get_resource_specification(model_name)
    time_vector = time_slice.time_base(resource.model_time_configuration)

    aws_keys = []
    for time in time_vector:
        aws_keys.append(
            _generate_aws_key(
                variable=variable,
                init_time=time[0],
                forecast_hour=time[1],
                model=resource)
        )
    return aws_keys


def _generate_aws_key(variable, init_time: datetime, forecast_hour: timedelta,
                      model: _RemoteResourceSpecification) -> str:
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

        if key == 'ecmwf_lead_zero_indicator':
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
    uri = model.uri_path_template(variable)
    for keyword, action in keywords.items():
        if keyword in uri:
            uri = action['method'](uri, keyword, *action['args'])

    return uri