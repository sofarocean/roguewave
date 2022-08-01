from datetime import datetime, timedelta
from boto3 import resource as boto3_resource
from functools import cache
from json import load
from os.path import expanduser
from roguewave.modeldata.timebase import ModelTimeConfiguration, \
    timebase_forecast, timebase_lead, timebase_evaluation
from typing import List, Tuple, Dict


# Model private variables
# =============================================================================

# Load the model configuration file.
try:
    with open(expanduser('~/model_configuration.json'), 'rb') as file:
        MODELS = load(file)
except FileNotFoundError as e:
    MODELS = {}


# Classes
# =============================================================================


class _RemoteResourceSpecification:
    """
    Class containing the key template and other necessary information to
    reconstruct uri's on the remote resource
    """

    def __init__(self,
                 name: str,
                 uri_path_template: str,
                 filetype: str = 'netcdf',
                 scheme: str = 's3',
                 mapping_variable_name_model_to_sofar: Dict[str,str] = None,
                 single_variable_per_file: bool = True,
                 model_time_configuration: dict = None):

        """
        :param name: Model name
        :param bucket: bucket
        :param uri_path_template: template of the AWS key
        :param filetype: filetype of the data
        :param model_time_configuration: Description on the timebase of the
        model (how often is the mode run, at what interval is output etc.).
        """
        self.name = name
        self.scheme = scheme
        self.uri_path_template = uri_path_template
        self.filetype = filetype
        self.mapping_variable_name_model_to_sofar = \
            mapping_variable_name_model_to_sofar
        self.single_variable_per_file = single_variable_per_file

        if model_time_configuration is None:
            self.model_time_configuration = ModelTimeConfiguration()
        else:
            self.model_time_configuration = ModelTimeConfiguration(
                **model_time_configuration
            )

    def to_sofar_variable_name(self,variable_name: str) -> str:
        """
        map model variable name to sofar variable name
        :param variable_name: model or sofar variable name
        :return: equivalent sofar_variable_name
        """
        if self.mapping_variable_name_model_to_sofar is not None:
            if variable_name in self.mapping_variable_name_model_to_sofar:
                return self.mapping_variable_name_model_to_sofar[
                    variable_name]
            else:
                # if not in the forward mapping, this may already be a sofar
                # variable name, see if this maps back to a model name, if so
                # this is a variable available in this model.
                try:
                    _ = self.to_model_variable_name(variable_name)
                    return variable_name

                except KeyError as e:
                    raise KeyError( f" Modem {self.name} does not have "
                                    f" output that corresponds to a variable "
                                    f" with name {variable_name}" )
        else:
            # if no variable definitions are available, we assume that this is
            # a sofar model, and the mapping is 1 to 1 i.e.
            #       model_variable_name = sofar_variable_name
            return variable_name

    def to_model_variable_name(self,variable_name: str) -> str:
        """
        map sofar variable name to model variable name
        :param variable_name: sofar or model variable name
        :return: equivalent model_variable_name
        """
        if self.mapping_variable_name_model_to_sofar is not None:
            for key,item in self.mapping_variable_name_model_to_sofar.items():
                if item==variable_name:
                    return key
                elif key==variable_name:
                    return key
            else:
                raise KeyError(f'no mapping for variable {variable_name} for model '
                               f'{self.name}')
        else:
            # if no variable definitions are available, we assume that this is
            # a sofar model, and the mapping is 1 to 1 i.e.
            #       model_variable_name = sofar_variable_name
            return variable_name


# Main Functions
# =============================================================================


def available_models() -> List[str]:
    """
    List all available models
    :return: Available model names as list
    """
    return list(MODELS.keys())


def model_timebase_forecast(
        model:str, init_time: datetime, duration: timedelta
        ) -> List[Tuple[datetime, timedelta]]:
    """
    Get all the valid times for a given forecast up to the requested duration.
    Valid times are returned as a List of pairs: (init_time, forecast_hour)

    :param model: name of the model to retrieve
    :param init_time: init time of the forecast
    :param duration: Duration requested
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """

    aws_layout = _get_resource_specification(model)
    return timebase_forecast(
        init_time=init_time,
        duration=duration,
        time_configuration=aws_layout.model_time_configuration
    )


# -----------------------------------------------------------------------------


def model_timebase_lead(
                  model:str,
                  start_time: datetime,
                  end_time: datetime,
                  lead_time: timedelta,
                  exact:bool = True
                  ) -> List[Tuple[datetime, timedelta]]:
    """
    Get all the valid times at a given lead-time in the period of interest.
    Valid times are returned as a List of pairs: (init_time, forecast_hour).
    If exact is true, only valid-times at exactly the requested lead-time are
    returned. Otherwise, it includes close valid_times as well.

    :param model: name of the model to retrieve
    :param start_time: start time of interval of interest
    :param end_time: end time of interval of interest
    :param lead_time:  lead time of interest
    :param exact: only return valid times that exactly match the requested lead
        time.
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """
    aws_layout = _get_resource_specification(model)
    return timebase_lead(
        start_time=start_time,
        end_time=end_time,
        lead_time=lead_time,
        time_configuration=aws_layout.model_time_configuration,
        exact=exact)


# -----------------------------------------------------------------------------


def model_timebase_evaluation(model:str,
                        evaluation_time: datetime,
                        maximum_lead_time: timedelta,
                        ) -> List[Tuple[datetime, timedelta]]:
    """
    Find all init times and forecast hours so that
        init_time + forecast = evaluation_time,

    :param model: name of the model to retrieve
    :param evaluation_time: Time we want to compare different forecasts for
    :param maximum_lead_time: Maximum lead time of the forecasts to consider
    :return: List of (init_time, forecast_hour) so that evaluation_time =
        init_time + forecast_hour
    """
    aws_layout = _get_resource_specification(model)
    return timebase_evaluation(
        evaluation_time=evaluation_time,
        maximum_lead_time=maximum_lead_time,
        time_configuration=aws_layout.model_time_configuration
    )


# -----------------------------------------------------------------------------
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
    uri = model.uri_path_template
    for keyword, action in keywords.items():
        if keyword in uri:
            uri = action['method'](uri, keyword, *action['args'])

    return uri


# -----------------------------------------------------------------------------


@cache
def list_available_variables(model_name, init_time: datetime = None):
    """
    List available variables from a given model. This function _only_ works
    if the variable names are encoded in the aws key. For e.g. ECMWF this
    currently does _not_ work.

    Basically, we look on AWS for the pattern given in model.key_template and
    find all keys conforming to the template with different values for the
    variable name. May be very slow if the prefix (everything before the
    variable name in the template) does not sufficiently restrict the search
    space.

    Note response is cached so that multiple calls are fast.

    :param model_name: the name of the model
    :param init_time: init_time, an init_time for which results from the model
    are available. Defaults to 2022-06-01; this may not work for all models.

    :return: list of variables for the model.
    """

    init_time = datetime(2022, 6, 1) if init_time is None else init_time
    # get model aws key template description
    model = _get_resource_specification(model_name)

    if model.mapping_variable_name_model_to_sofar is not None:
        return list(model.mapping_variable_name_model_to_sofar.keys())

    # open s3 resource
    s3 = boto3_resource('s3')
    my_bucket = s3.Bucket(model.bucket)

    if 'variable' not in model.uri_path_template:
        raise ValueError('Listing variables for a particular model only works'
                         'if the variable name is part of the aws keys.')

    # replace "variable" with "####" as a seperator between everything before
    # and after the variable name in the aws key.
    key = _generate_aws_key(
        '####', init_time, forecast_hour=timedelta(hours=0), model=model)

    # Split in prefix and suffix, and remove the bucket name from the prefix
    prefix, suffix = key.split('####')
    prefix = prefix.replace(model.bucket + '/', '')

    # loop over all objects conforming to the prefix
    variables = []
    for obj in my_bucket.objects.filter(Prefix=prefix):
        if suffix in obj.key:
            # If the suffix is part of the key, add the part in between the
            # prefix and the suffix of the aws key as a variable name to the
            # output list.
            variables.append(
                _find_between(obj.key, prefix, suffix)
            )

    return variables


# Helper functions
# =============================================================================


def _get_resource_specification(name) -> _RemoteResourceSpecification:
    """
    Return the object representing key layout on aws
    :param name:
    :return: Model
    """
    return _RemoteResourceSpecification(name=name, **MODELS[name])


# -----------------------------------------------------------------------------


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
