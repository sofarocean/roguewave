from datetime import datetime, timedelta
from boto3 import resource as boto3_resource
from functools import cache
from json import load
from os.path import expanduser
from roguewave.modeldata.timebase import ModelTimeConfiguration, \
    TimeSlice
from typing import List, Dict, Tuple, Literal, Mapping


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
model_type = Literal['analysis','forecast']

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
                 model_type: model_type = 'analysis',
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
        self.model_type = model_type
        if not isinstance(uri_path_template,Mapping):
            self._uri_path_template = {"default":uri_path_template}
        else:
            self._uri_path_template = uri_path_template
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

    def uri_path_template(self, variable):
        if variable not in self._uri_path_template:
            return self._uri_path_template['default']
        else:
            return self._uri_path_template[variable]


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


def model_timebase( time_slice:TimeSlice, model:str
                    ) -> List[Tuple[datetime, timedelta]]:
    """
    Get all the valid times for a given time slice for the requested model.
    Valid times are returned as a List of pairs: (init_time, forecast_hour)

    :param time_slice: time slice
    :param model: name of the model to retrieve
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """
    resource = _get_resource_specification(model)
    return time_slice.time_base(resource.model_time_configuration)


def model_valid_time( time_slice:TimeSlice, model:str) -> List[datetime]:
    """
    Get all the valid times for a given time slice for the requested model.
    Valid times are returned as a List of pairs: (init_time, forecast_hour)

    :param time_slice: time slice
    :param model: name of the model to retrieve
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """
    timebase = model_timebase(time_slice,model)
    return [ x[0] + x[1] for x in timebase ]

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

    # This is ugly, but avoids a circular import. ToDo: redo this function.
    from roguewave.modeldata.keygeneration import _generate_aws_key

    init_time = datetime(2022, 6, 1) if init_time is None else init_time
    # get model aws key template description
    model = _get_resource_specification(model_name)

    if model.mapping_variable_name_model_to_sofar is not None:
        return list(model.mapping_variable_name_model_to_sofar.keys())

    # open s3 resource
    s3 = boto3_resource('s3')


    if 'variable' not in model.uri_path_template('default'):
        raise ValueError('Listing variables for a particular model only works'
                         'if the variable name is part of the aws keys.')

    # replace "variable" with "####" as a seperator between everything before
    # and after the variable name in the aws key.
    key = _generate_aws_key(
        '####', init_time, forecast_hour=timedelta(hours=0), model=model)

    # Split in prefix and suffix, and remove the bucket name from the prefix
    prefix, suffix = key.split('####')
    prefix = prefix.replace('s3://','')
    bucket,prefix = prefix.split('/',1)


    # loop over all objects conforming to the prefix
    variables = []
    my_bucket = s3.Bucket(bucket)
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
