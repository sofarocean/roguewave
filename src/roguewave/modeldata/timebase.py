from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Union,Tuple

INTERVALTYPE = Union[ List[tuple[int,timedelta]], Tuple[tuple[int,timedelta]]]

@dataclass()
class ModelTimeConfiguration():
    cycle_time_hours: timedelta = timedelta(hours=6)
    cycle_offset_hours: timedelta = timedelta(hours=0)
    output_interval: INTERVALTYPE = ( ( 240,timedelta(hours=1) ), )
    _output_interval: INTERVALTYPE =  field(init=False, repr=False)

    @property
    def output_interval(self) -> INTERVALTYPE:
        return self._output_interval

    @output_interval.setter
    def output_interval(self, values):
        self._output_interval = []
        for value in values:
            if isinstance(value[1], timedelta):
                self._output_interval.append( value)
            else:
                self._output_interval.append((value[0],timedelta(hours=value[1])))

def timebase_forecast( init_time:datetime,duration:timedelta,time_configuration:ModelTimeConfiguration):
    timebase = []
    for intervals in time_configuration.output_interval:
        #
        for index in range(intervals[0]):
            timebase.append( ( init_time, intervals[1] * index ) )

            if intervals[1] * index >= duration:
                return timebase

    return timebase

def timebase_lead( start_time:datetime,lead_time:timedelta,time_configuration:ModelTimeConfiguration,exact=True):
    pass

def timebase_evaluation( start_time:datetime, end_time:datetime,time_configuration:ModelTimeConfiguration ):
    pass

