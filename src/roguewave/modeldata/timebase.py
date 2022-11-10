"""
Contents: Routines to get data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to calculate times we have output at different init and forecasthours

Functions:

- `find_clostest_init_time`, find closest model init time to a given valid time
- `timebase forecast`, get valid forecast times
- `timebase_lead`, get all output from the model at a given lead time.
- `timebase_eval`, get all output from different forecasts at a given
    evaluation time.
"""


# Import
# =============================================================================

from datetime import datetime, timezone, timedelta
from typing import List, Union, Tuple
from roguewave.tools.time import to_datetime_utc


# Model private variables
# =============================================================================


INTERVAL_TYPE = Union[List[tuple[int, timedelta]], Tuple[tuple[int, timedelta]]]

# Classes
# =============================================================================


class ModelTimeConfiguration:
    def __init__(
        self,
        cycle_time_hours: timedelta = timedelta(hours=6),
        cycle_offset_hours: timedelta = timedelta(hours=0),
        output_interval: INTERVAL_TYPE = ((239, timedelta(hours=1)),),
    ):

        self.cycle_time_hours = cycle_time_hours
        self.cycle_offset_hours = cycle_offset_hours
        self.output_interval = output_interval
        self.duration = self.forecast_hours()[-1]

    @property
    def cycle_time_hours(self):
        return self._cycle_time_hours

    @cycle_time_hours.setter
    def cycle_time_hours(self, value):
        self._cycle_time_hours = _to_timedelta(value)

    @property
    def cycle_offset_hours(self):
        return self._cycle_offset_hours

    @cycle_offset_hours.setter
    def cycle_offset_hours(self, value):
        self._cycle_offset_hours = _to_timedelta(value)

    @property
    def output_interval(self) -> INTERVAL_TYPE:
        return self._output_interval

    @output_interval.setter
    def output_interval(self, values):
        self._output_interval = []
        for value in values:
            if isinstance(value[1], timedelta):
                self._output_interval.append(value)
            else:
                self._output_interval.append((value[0], timedelta(hours=value[1])))

    def forecast_hours(
        self, duration: timedelta = None, endpoint=False
    ) -> List[timedelta]:
        forecast_hours = [timedelta(hours=0)]
        for interval in self.output_interval:
            base_hour = forecast_hours[-1]
            for index in range(interval[0]):
                forecast_hours.append(base_hour + interval[1] * (index + 1))
                if duration is not None:
                    if endpoint:
                        if forecast_hours[-1] > duration:
                            return forecast_hours
                    else:
                        if forecast_hours[-1] >= duration:
                            return forecast_hours
        return forecast_hours

    def interval_at_lead_time(self, lead_time: timedelta) -> timedelta:
        """
        Get the output interval at a certain lead time
        :param lead_time:
        :return: output interval at the given lead time
        """
        if lead_time > self.duration:
            raise ValueError("Lead time exceeds duration of the forecast")

        forecast_hours = [timedelta(hours=0)]
        for interval in self.output_interval:
            base_hour = forecast_hours[-1]
            if lead_time <= base_hour + interval[0] * interval[1]:
                interval = interval[1]
                break
        else:
            interval = self.output_interval[-1][1]
        return interval


class TimeSlice:
    def __init__(self, start_time: datetime, end_time: datetime, endpoint=False):
        self.start_time = to_datetime_utc(start_time)
        self.end_time = to_datetime_utc(end_time)
        self.endpoint = endpoint

    def time_base(
        self, time_configuration: ModelTimeConfiguration
    ) -> List[Tuple[datetime, timedelta]]:
        pass


class TimeSliceForecast(TimeSlice):
    def __init__(self, init_time: datetime, duration: timedelta, endpoint=False):
        """
        Timeslice for a single forecast
        :param init_time: init time of forecast (datetime)
        :param duration: forecast length (time delta)
        """
        self.init_time = to_datetime_utc(init_time)
        self.duration = duration
        super().__init__(self.init_time, self.init_time + duration, endpoint)

    def time_base(self, time_configuration: ModelTimeConfiguration):
        return timebase_forecast(
            init_time=self.init_time,
            duration=self.duration,
            time_configuration=time_configuration,
            endpoint=self.endpoint,
        )


class TimeSliceLead(TimeSlice):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        lead_time: timedelta,
        exact=False,
        endpoint=False,
    ):
        """
        Slice at constant lead time
        :param start_time: start time of period of interest (datetime)
        :param end_time: end time of period of interest (datetime)
        :param lead_time: lead time (timedelta) of interest.
        :param exact: boolean. Do we want data only exactly at the given lead
            time, or as close as possible
        """
        super().__init__(start_time, end_time, endpoint)
        self.lead_time = lead_time
        self.exact = exact

    def time_base(self, time_configuration: ModelTimeConfiguration):
        return timebase_lead(
            start_time=self.start_time,
            end_time=self.end_time,
            lead_time=self.lead_time,
            time_configuration=time_configuration,
            exact=self.exact,
            endpoint=self.endpoint,
        )


class TimeSliceAnalysis(TimeSlice):
    def __init__(self, start_time: datetime, end_time: datetime, endpoint=False):
        """
        Analysis time
        :param start_time: start time of period of interest (datetime)
        :param end_time: end time of period of interest (datetime)
        """
        super().__init__(start_time, end_time, endpoint)

    def time_base(self, time_configuration: ModelTimeConfiguration):
        return timebase_lead(
            start_time=self.start_time,
            end_time=self.end_time,
            lead_time=timedelta(hours=0),
            time_configuration=time_configuration,
            exact=False,
            endpoint=self.endpoint,
        )


class TimeSliceBestForecast(TimeSliceAnalysis):
    def __init__(self, start_time: datetime, end_time: datetime, endpoint=False):
        """
        Analysis time
        :param start_time: start time of period of interest (datetime)
        :param end_time: end time of period of interest (datetime)
        """
        super().__init__(start_time, end_time, endpoint)

    def time_base(self, time_configuration: ModelTimeConfiguration):
        return super().time_base(time_configuration)


class TimeSliceEvaluation(TimeSlice):
    def __init__(
        self, evaluation_time: datetime, maximum_lead_time: timedelta, endpoint=False
    ):
        """
        All data for a constant point in time
        :param evaluation_time: evaluation point (datetime)
        :param maximum_lead_time:  maximum lead time considered (timedelta)
        """
        super().__init__(
            start_time=evaluation_time, end_time=evaluation_time, endpoint=endpoint
        )
        self.evaluation_time = evaluation_time
        self.maximum_lead_time = maximum_lead_time

    def time_base(self, time_configuration: ModelTimeConfiguration):
        return timebase_evaluation(
            evaluation_time=self.evaluation_time,
            maximum_lead_time=self.maximum_lead_time,
            time_configuration=time_configuration,
            endpoint=self.endpoint,
        )


# Main Public Functions
# =============================================================================
def find_closet_init_time(
    evaluation_time: datetime,
    cycle_time_hours: timedelta = timedelta(hours=6),
    cycle_offset_hours: timedelta = timedelta(hours=0),
    lead_time: timedelta = timedelta(hours=0),
) -> datetime:
    """
    Find the model init time that is closest to the evaluation time requested
    so that init_time <= evaluation_time - lead_time

    :param evaluation_time: Time of interest
    :param cycle_time_hours:  cycle time of the model
    :param cycle_offset_hours: offset of the model
    :param lead_time: Lead time of interest
    :return: init time
    """
    evaluation_time_seconds = evaluation_time.timestamp()
    cycle_time_seconds = cycle_time_hours.total_seconds()
    cycle_offset_seconds = cycle_offset_hours.total_seconds()
    lead_time_seconds = lead_time.total_seconds()

    # Find the nearest cycle number.
    number_of_cycles = (
        evaluation_time_seconds - cycle_offset_seconds - lead_time_seconds
    ) // cycle_time_seconds

    init_time = datetime.fromtimestamp(
        number_of_cycles * cycle_time_seconds + cycle_offset_seconds, tz=timezone.utc
    )

    # The delta represents the actual lead time of the evaluation time
    # compared to the init time returned
    delta = evaluation_time_seconds - lead_time_seconds - init_time.timestamp()
    if delta >= cycle_time_seconds / 2:
        # If the delta is larger than half a cycle, a cycle with a longer lead
        # time is actually closer to the desired lead time. For example if
        # the requested lead_time is 6 hours, the cycle time 6 hours and the
        # actual lead time for init_time t is one hours, it makes sense to
        # return t0 - 6 hours as the init time as that has a lead time of 7
        # hours, which is closer to the desired lead time of 6 hours, and
        # consequently more representative of the desired forecast accuracy.
        new_lead_time = evaluation_time - init_time - cycle_time_hours
        if new_lead_time.total_seconds() >= 0:
            init_time = init_time + cycle_time_hours

    return init_time


def timebase_forecast(
    init_time: datetime,
    duration: timedelta,
    time_configuration: ModelTimeConfiguration,
    endpoint=False,
) -> List[Tuple[datetime, timedelta]]:
    """
    Get all the valid times for a given forecast up to the requested duration.
    Valid times are returned as a List of pairs: (init_time, forecast_hour)

    :param init_time: init time of the forecast
    :param duration: Duration requested
    :param time_configuration:  Output time configuration of the model
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """

    return [
        (init_time, forecast_hour)
        for forecast_hour in time_configuration.forecast_hours(
            duration=duration, endpoint=endpoint
        )
    ]


def timebase_lead(
    start_time: datetime,
    end_time: datetime,
    lead_time: timedelta,
    time_configuration: ModelTimeConfiguration,
    exact=True,
    endpoint=False,
) -> List[Tuple[datetime, timedelta]]:
    """
    Get all the valid times at a given lead-time in the period of interest.
    Valid times are returned as a List of pairs: (init_time, forecast_hour).
    If exact is true, only valid-times at exactly the requested lead-time are
    returned. Otherwise, it includes close valid_times as well.

    :param start_time: start time of interval of interest
    :param end_time: end time of interval of interest
    :param lead_time:  lead time of interest
    :param time_configuration: Output time configuration of the model
    :param exact: only return valid times that exactly match the requested lead
        time.
    :return: List of (init_time, forecast_hour) so that valid_time =
        init_time + forecast_hour
    """

    # Make sure the lead time does not exceed the duration
    if lead_time > time_configuration.duration:
        raise ValueError(
            f"Lead time {lead_time} exceeds maximum forecast "
            f"length of {time_configuration.duration}"
        )

    forecast_hours = time_configuration.forecast_hours(endpoint=endpoint)

    # If exact, make sure the lead time is part of the forecast.
    if exact and (lead_time not in forecast_hours):
        raise ValueError(
            f"Forecast does not have output " f"exactly at lead time: {lead_time}"
        )

    # Get the output interval at the requested lead time
    interval = time_configuration.interval_at_lead_time(lead_time=lead_time)

    # Calculate number of instances we will return (upper bound)
    n = int((end_time - start_time) / interval) + int(endpoint)

    timebase = []
    for index in range(0, n):
        #
        valid_time = start_time + index * time_configuration.output_interval[0][1]

        init_time = find_closet_init_time(
            valid_time,
            cycle_time_hours=time_configuration.cycle_time_hours,
            cycle_offset_hours=time_configuration.cycle_offset_hours,
            lead_time=lead_time,
        )

        forecast_hour = valid_time - init_time

        # if exact=False, and we are on the edge between two different output
        # intervals not all of the forecast hours returned may exist.
        if forecast_hour not in forecast_hours:
            continue

        # If exact, only add at exact lead times.
        if exact and not (forecast_hour == lead_time):
            continue

        timebase.append((init_time, forecast_hour))

    return timebase


def timebase_evaluation(
    evaluation_time: datetime,
    maximum_lead_time: timedelta,
    time_configuration: ModelTimeConfiguration,
    endpoint=False,
) -> List[Tuple[datetime, timedelta]]:
    """
    Find all init times and forecast hours so that
        init_time + forecast = evaluation_time,

    :param evaluation_time: Time we want to compare different forecasts for
    :param maximum_lead_time: Maximum lead time of the forecasts to consider
    :param time_configuration: Output time configuration of the model.
    :return: List of (init_time, forecast_hour) so that evaluation_time =
        init_time + forecast_hour
    """

    forecast_hours = time_configuration.forecast_hours(
        duration=maximum_lead_time, endpoint=endpoint
    )

    timebase = []
    for forecast_hour in forecast_hours:
        init_time = find_closet_init_time(
            evaluation_time=evaluation_time,
            cycle_time_hours=time_configuration.cycle_time_hours,
            cycle_offset_hours=time_configuration.cycle_offset_hours,
            lead_time=forecast_hour,
        )
        if evaluation_time == init_time + forecast_hour:
            timebase.append((init_time, forecast_hour))

    return timebase


# Helper functions
# =============================================================================


def _to_timedelta(value):
    if isinstance(value, timedelta):
        return value
    else:
        return timedelta(hours=int(value))
