from datetime import datetime, timedelta
from typing import List
import boto3
from roguewave.modeldata.keygeneration import MODELS, \
    _generate_aws_key, _find_between, _get_model_aws_layout
from functools import cache


def available_models() -> List[str]:
    """
    List all available models
    :return: Available model names as list
    """
    return list(MODELS.keys())


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
    model = _get_model_aws_layout(model_name)

    # open s3 resource
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(model.bucket)

    if 'variable' not in model.key_template:
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
