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

class Model():
    def __init__(self, name, bucket:str, key_template:str, filetype:str='netcdf',model_time_configuration:dict=None):
        self.name = name
        self.bucket = bucket
        self.key_template = key_template
        self.filetype = filetype
        if model_time_configuration is None:
            self.model_time_configuration = ModelTimeConfiguration()
        else:
            self.model_time_configuration = ModelTimeConfiguration(**model_time_configuration)

def get_model(name):
    return Model(name=name,**MODELS[name])

def generate_aws_key(variable, init_time: datetime, forecast_hour: timedelta,
                     model: Model):

    def _find_between(s, first, last):
        # lets avoid regex for now :-)
        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def _replace(string,key,value):
        return string.replace("{" + key + "}", value)

    def _time(string,key,value:datetime):
        while key in string:
            format = _find_between(string,'{'+key+':','}')
            to_replace = key+':'+format
            replace_with= value.strftime(format)
            string = _replace( string,to_replace, replace_with)
        return string

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
        "bucket": { 'method':_replace, 'args':(model.bucket,)},
        "forecasthour": { 'method':_replace, 'args':("{hours:03d}".format(hours=int(forecast_hour.total_seconds() // 3600)),)},
        "init_time": { 'method':_time, 'args':(init_time,)},
        "valid_time": {'method': _time, 'args': (valid_time,)},
        "variable": { 'method':_replace, 'args':(variable,)},
        "ecmwf_subcycle_string":{'method':_replace, 'args':(ecmwf_subcycle_string,)},
        "ecmwf_lead_zero_indicator":{'method':_replace, 'args':(ecmwf_lead_zero_indicator,)}
    }

    string = model.key_template
    for key, value in properties.items():
        if key in string:
            string = value['method'](string,key,*value['args'])

    return f"{model.bucket}/{string}"


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

def generate_evaluation_time_keys_and_valid_times(variable, start_time: datetime, end_time: datetime,
                            lead_time: timedelta, model: Model, exact=False) -> Tuple[List[str],List[datetime]]:

    pass
