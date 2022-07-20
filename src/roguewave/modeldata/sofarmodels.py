from dataclasses import dataclass
from datetime import datetime, timedelta
from .timebase import ModelTimeConfiguration, timebase_forecast, timebase_lead, \
    timebase_evaluation
import os
from typing import List,Tuple
import json

with open( os.path.expanduser('~/model_configuration.json'),'rb') as file:
    MODELS = json.load(file)


def print_available_models():
    for key in MODELS:
        print(key)

@dataclass()
class Model():
    key:str
    name: str
    label: str
    bucket: str
    resolution: str
    region: str
    prefix: str
    suffix: str
    model_time_configuration: dict


def get_model(name):
    return Model(key=name,**MODELS[name])


def _replace(string: str, properties):
    for key, value in properties.items():
        if key in string:
            string = string.replace("{" + key + "}", value)
    return string


def generate_aws_key(variable, init_time: datetime, forecast_hour: timedelta,
                     model: Model):
    #
    properties = {
        "name": model.name,
        "label": model.label,
        "bucket": model.bucket,
        "resolution": model.resolution,
        "region": model.region,
        "date": init_time.strftime("%Y%m%d"),
        "cyclehour": init_time.strftime("%H"),
        "forecasthour": "{hours:03d}".format(
            hours=int(forecast_hour.total_seconds() // 3600)),
        "variable": variable,
        "key":model.key
    }
    prefix = _replace(model.prefix, properties)
    suffix = _replace(model.suffix, properties)

    return f"{model.bucket}/{prefix}/{suffix}"


def generate_forecast_keys(variable, init_time: datetime, duration: timedelta,
                           model: Model) -> Tuple[List[str],List[datetime]] :
    model_time_configuration = ModelTimeConfiguration(
        **model.model_time_configuration)

    time_vector = timebase_forecast(init_time, duration,
                                    time_configuration=model_time_configuration)

    aws_keys = []
    valid_time = []
    for time in time_vector:
        aws_keys.append(
            generate_aws_key(variable, init_time=time[0],
                             forecast_hour=time[1], model=model)
        )
        valid_time.append(time[0]+time[1])
    return aws_keys, valid_time


def generate_lead_keys(variable, start_time: datetime, lead_time: timedelta,
                       model: Model, exact=True) -> List[str]:
    model_time_configuration = ModelTimeConfiguration(
        **model.model_time_configuration)

    time_vector = timebase_lead(start_time, lead_time,
                                time_configuration=model_time_configuration,
                                exact=exact)

    aws_keys = []
    for time in time_vector:
        aws_keys.append(
            generate_aws_key(variable, init_time=time[0],
                             forecast_hour=time[1], model=model)
        )
    return aws_keys
