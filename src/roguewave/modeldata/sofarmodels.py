from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .timebase import ModelTimeConfiguration, timebase_forecast, timebase_lead, \
    timebase_evaluation
import os
from typing import List,Tuple
import json

with open( os.path.expanduser('~/model_configuration.json'),'rb') as file:
    MODELS = json.load(file)

def available_models():
    return list(MODELS.keys())

@dataclass()
class Model():
    key:str
    name: str
    label: str
    bucket: str
    resolution: str
    region: str
    key_template: str
    filetype:str
    model_time_configuration: ModelTimeConfiguration
    _model_time_configuration: ModelTimeConfiguration =  field(init=False, repr=False)

    @property
    def model_time_configuration(self) -> ModelTimeConfiguration:
        return self._model_time_configuration

    @model_time_configuration.setter
    def model_time_configuration(self, values):
        self._model_time_configuration = ModelTimeConfiguration(**values)


def get_model(name):
    return Model(key=name,**MODELS[name])


def _replace(string: str, properties):
    for key, value in properties.items():
        if key in string:
            string = string.replace("{" + key + "}", value)
    return string

#ecmwf/{prefix}{date_string}{hour_string}00{valid_time_date}{valid_time_hour}011
def generate_aws_key(variable, init_time: datetime, forecast_hour: timedelta,
                     model: Model):
    #
    if forecast_hour == timedelta(hours=0):
        ecmwf_lead_zero_indicator = '011'
    else:
        ecmwf_lead_zero_indicator = '001'

    if (init_time.hour == 0) or (init_time.hour == 12):
        ecmwf_subcycle_string = 'A2P'
    else:
        ecmwf_subcycle_string = 'A2Q'
    valid_time = init_time+forecast_hour

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
        "key":model.key,
        "init_time_month": init_time.strftime("%m"),
        "init_time_day": init_time.strftime("%d"),
        "valid_time_month": valid_time.strftime("%m"),
        "valid_time_day": valid_time.strftime("%d"),
        "valid_time_hour": valid_time.strftime("%H"),
        "ecmwf_subcycle_string":ecmwf_subcycle_string,
        "ecmwf_lead_zero_indicator":ecmwf_lead_zero_indicator
    }
    key = _replace(model.key_template, properties)

    return f"{model.bucket}/{key}"


def generate_forecast_keys_and_valid_times(variable, init_time: datetime, duration: timedelta,
                           model: Model) -> Tuple[List[str],List[datetime]] :

    time_vector = timebase_forecast(init_time, duration,
                                    time_configuration=model.model_time_configuration)

    aws_keys = []
    valid_time = []
    for time in time_vector:
        aws_keys.append(
            generate_aws_key(variable, init_time=time[0],
                             forecast_hour=time[1], model=model)
        )
        valid_time.append(time[0]+time[1])
    return aws_keys, valid_time


def generate_lead_keys_and_valid_times(variable, start_time: datetime, end_time: datetime,
                            lead_time: timedelta, model: Model, exact=False) -> Tuple[List[str],List[datetime]]:

    time_vector = timebase_lead(start_time, end_time, lead_time,
                                time_configuration=model.model_time_configuration,
                                exact=exact)

    aws_keys = []
    valid_time = []
    for time in time_vector:
        aws_keys.append(
            generate_aws_key(variable, init_time=time[0],
                             forecast_hour=time[1], model=model)
        )
        valid_time.append(time[0] + time[1])
    return aws_keys, valid_time
