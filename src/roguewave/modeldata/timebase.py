from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Union, Tuple
import numpy

INTERVALTYPE = Union[List[tuple[int, timedelta]], Tuple[tuple[int, timedelta]]]


def to_timedelta(value):
    if isinstance(value, timedelta):
        return value
    else:
        return timedelta(hours=int(value))


def find_closet_init_time(evaluation_time: datetime,
                          cycle_time_hours: timedelta,
                          cycle_offset_hours: timedelta,
                          lead_time: timedelta = timedelta(hours=0)):
    evaluation_time_seconds = evaluation_time.timestamp()
    cycle_time_seconds = cycle_time_hours.total_seconds()
    cycle_offset_seconds = cycle_offset_hours.total_seconds()
    lead_time_seconds = lead_time.total_seconds()

    number_of_cycles = (evaluation_time_seconds - cycle_offset_seconds- lead_time_seconds) // \
                       cycle_time_seconds



    init_time = datetime.fromtimestamp(
        number_of_cycles * cycle_time_seconds
            + cycle_offset_seconds,
        tz=timezone.utc
    )
    delta = evaluation_time_seconds - lead_time_seconds - init_time.timestamp()

    if delta >= cycle_time_seconds/2:
        new_lead_time = evaluation_time - init_time - cycle_time_hours
        if new_lead_time.total_seconds() >= 0:
            init_time = init_time + cycle_time_hours

    #print( (lead_time_seconds - cycle_time_seconds * (lead_time_seconds//cycle_time_seconds))/3600 //2)
    #
    # actual_lead = evaluation_time - init_time
    # lead_delta  = int(lead_time.total_seconds()-actual_lead.total_seconds()/3600)
    #
    # number_of_cycles = numpy.round(lead_delta/cycle_time_seconds,0)
    # init_time = init_time-number_of_cycles*cycle_time_hours
    return init_time


class ModelTimeConfiguration():
    def __init__(self,
                 cycle_time_hours: timedelta = timedelta(hours=6),
                 cycle_offset_hours: timedelta = timedelta(hours=0),
                 output_interval: INTERVALTYPE = (
                         (240, timedelta(hours=1)),)):

        self.cycle_time_hours = cycle_time_hours
        self.cycle_offset_hours = cycle_offset_hours
        self.output_interval = output_interval

    @property
    def cycle_time_hours(self):
        return self._cycle_time_hours

    @cycle_time_hours.setter
    def cycle_time_hours(self, value):
        self._cycle_time_hours = to_timedelta(value)

    @property
    def cycle_offset_hours(self):
        return self._cycle_offset_hours

    @cycle_offset_hours.setter
    def cycle_offset_hours(self, value):
        self._cycle_offset_hours = to_timedelta(value)

    @property
    def output_interval(self) -> INTERVALTYPE:
        return self._output_interval

    @output_interval.setter
    def output_interval(self, values):
        self._output_interval = []
        for value in values:
            if isinstance(value[1], timedelta):
                self._output_interval.append(value)
            else:
                self._output_interval.append(
                    (value[0], timedelta(hours=value[1])))


def timebase_forecast(init_time: datetime, duration: timedelta,
                      time_configuration: ModelTimeConfiguration):
    timebase = [(init_time, timedelta(hours=0))]
    for intervals in time_configuration.output_interval:
        #
        base_hour = timebase[-1][1]
        for index in range(intervals[0]):
            timebase.append(
                (init_time, base_hour + intervals[1] * (index + 1)))

            if intervals[1] * index >= duration:
                return timebase

    return timebase


def timebase_lead(start_time: datetime, end_time: datetime,lead_time: timedelta,
                  time_configuration: ModelTimeConfiguration, exact=True):

    n = int((end_time-start_time)/time_configuration.output_interval[0][1])
    times = []
    for index in range(0, n):
        #
        valid_time = start_time + index * time_configuration.output_interval[0][1]
        init_time = find_closet_init_time(valid_time,
                                      cycle_time_hours=time_configuration.cycle_time_hours,
                                      cycle_offset_hours=time_configuration.cycle_offset_hours,
                                      lead_time=lead_time)
        forecast_hours = valid_time-init_time
        if exact:
            if not (forecast_hours==lead_time):
                continue

        times.append( (init_time,forecast_hours) )

    return times


def timebase_evaluation(start_time: datetime, end_time: datetime,
                        time_configuration: ModelTimeConfiguration):
    pass
