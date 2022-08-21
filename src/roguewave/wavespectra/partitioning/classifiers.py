from datetime import datetime, timezone, timedelta
from pandas import DataFrame
from typing import Union, List, Tuple, TypedDict
import numpy

def get_overlapped_section(model_time: numpy.ndarray,
                           model_variable: numpy.ndarray,
                           observed_time: numpy.ndarray,
                           observed_variable: numpy.ndarray,
                           time_buffer: timedelta = None,
                           wrapping_angle: float = None) -> Union[
    None, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]:
    """
    A function that takes two timeseries, a "model" and "observed" series, and
    interpolates them to the same overlapping time interval. Anything outside
    time overlap is discarded unless a buffer is applied- in which case a buffer
    will be added to the left and right of the overlapping interval with missing
    values set to NaN.

    :param model_time: Numpy array of datetime objects
    :param model_variable: 1D numpy array of model values
    :param observed_time: Numpy array of datetime objects
    :param observed_variable: 1D numpy array of observed values
    :param time_buffer: buffer
    :param wrapping_angle: if wrapped variable (e.g. direction) this is the wrapping period.
    :return: If there is overlap: a tuple of three numpy arrays: (
                 time vector (datetime, utc) of overlapping interval,
                 interpolated model values,
                 interpolated time values
                 )
             else:
                 None
    """

    # Pandas does not support negative indices. To ensure this works with
    # pandas series as input use length
    last_element_observed = len(observed_time) - 1
    last_element_modeled = len(model_time) - 1

    # No overlap, return None
    if model_time[0] > observed_time[last_element_observed] or model_time[
        last_element_modeled] < observed_time[0]:
        return None

    # Convert to unix epoch timestamps for interpolation
    model_time = numpy.array([x.timestamp() for x in model_time])
    observed_time = numpy.array([x.timestamp() for x in observed_time])

    # get the start and end time of the overlapping interval
    time_start = model_time[0] if model_time[0] >= observed_time[0] else \
        observed_time[0]
    time_end = model_time[last_element_modeled] if model_time[
                                                       last_element_modeled] <= \
                                                   observed_time[
                                                       last_element_observed] else \
        observed_time[last_element_observed]

    # get the minimum timestep in the output time vector
    delta = min(numpy.min(numpy.diff(model_time)),
                numpy.min(numpy.diff(observed_time)))

    time_end = (int((time_end - time_start) / delta) + 1) * delta + time_start

    # If desired, add a buffer to the left and right of the output interval
    if time_buffer:
        buffer_seconds = delta * (int(time_buffer.total_seconds() / delta))
        time_start -= buffer_seconds
        time_end += buffer_seconds

    # Get the time vector
    time = numpy.linspace(time_start, time_end,
                          int((time_end - time_start) / delta + 1),
                          endpoint=True)

    # If this is a wrapped variable, first unwrap before interpolation
    if wrapping_angle is not None:
        model_variable = numpy.unwrap(model_variable, discont=wrapping_angle)
        observed_variable = numpy.unwrap(observed_variable,
                                         discont=wrapping_angle)

    # Interpolate onto the overlapping time interval
    model_value = numpy.interp(time, model_time,
                               model_variable,
                               left=numpy.nan, right=numpy.nan)
    observed_value = numpy.interp(time, observed_time,
                                  observed_variable,
                                  left=numpy.nan, right=numpy.nan)

    # Convert time back to datetime objects, with UTC timezone
    time_utc = numpy.array(
        [datetime.fromtimestamp(x, tz=timezone.utc) for x in time])

    # If this is a wrapped variable, apply modulo to project back to
    # [0, wrapping_angle] interval
    if wrapping_angle is not None:
        model_value = model_value % wrapping_angle
        observed_value = observed_value % wrapping_angle

    # Return
    return time_utc, model_value, observed_value


def partitions_overlap_in_time(model_time, observed_time) -> bool:
    last_element_observed = len(observed_time) - 1
    last_element_modeled = len(model_time) - 1

    # No overlap, return None
    if model_time[0] > observed_time[last_element_observed] or model_time[
        last_element_modeled] < observed_time[0]:
        return True
    else:
        return False


class MatchOutput(TypedDict):
    model_index: int
    observed_index: int
    model_field: DataFrame
    observed_field: DataFrame
    model_lag: timedelta
    fit: dict


def match(model_fields: List[DataFrame],
          observed_fields: List[DataFrame],
          time_buffer: timedelta = timedelta(hours=24),
          average_time: timedelta = timedelta(hours=24),
          period_threshold=0.8,
          waveheight_threshold=0.8,
          direction_threshold=0.8) -> \
        List[MatchOutput]:
    #
    """
    Try to match wavefields from observed partitions to modeled partitions.
    :param model_fields:
    :param observed_fields:
    :param time_buffer:
    :return:
    """

    # Nested Dict to store matches, first key is the model field index, second key
    # the observed index and the contents are some metrics on the lag and
    # goodness of fit
    matched_to_observation = {}  # [{} for x in observed_fields]

    # Loop over all combinations. To note; fits are _not_ symmetric.
    output = []
    for model_index, model in enumerate(model_fields):
        #
        for observed_index, obs in enumerate(observed_fields):
            #
            # Find the minimum lag between model and observation
            #
            lag = find_model_lag(
                model['time'],
                [model['hm0'], model['tm01'], model['mean_direction']],
                obs['time'],
                [obs['hm0'], obs['tm01'], obs['mean_direction']],
                time_buffer,
                wrapping_angle=[None, None, 360]
            )
            if lag is None:
                # if the lag is None the sections do not overlap
                continue

            variables = ['hm0', 'tm01', 'mean_direction']
            wrapping_angles = [None, None, 360]
            fit = {}
            # Calculate the fit value for waveheight, period and direction once
            # corrected for potential lag
            for variable, wrapping_angle in zip(variables, wrapping_angles):
                time_utc, model_value, observed_value = get_overlapped_section(
                    model['time'] - lag,
                    model[variable],
                    obs['time'],
                    obs[variable],
                    time_buffer=time_buffer,
                    wrapping_angle=wrapping_angle)


                time_delta = time_utc[1] - time_utc[0]
                fit[variable] = time_delta * fit_quality(
                    model_value,observed_value,wrapping_angle) / average_time

            # If the fit value is good enough- we consider it a match
            if fit['hm0'] > waveheight_threshold and fit[
                'tm01'] > period_threshold and fit[
                'mean_direction'] > direction_threshold:

                output.append(MatchOutput(
                    model_index=model_index,
                    observed_index=observed_index,
                    model_field=model_fields[model_index],
                    observed_field=observed_fields[observed_index],
                    model_lag=lag,
                    fit=fit
                ))
    return output


def fit_quality(x, y, wrapping_angle=None):
    if wrapping_angle is not None:
        diff = numpy.abs(
            (x - y + wrapping_angle / 2) % wrapping_angle
            - wrapping_angle / 2
        )
        scale = wrapping_angle / 4
    else:
        diff = numpy.abs(x - y)
        scale = numpy.minimum(x, y)

    return numpy.nansum(1 - numpy.tanh(2 * diff / scale))


def find_model_lag(model_time, model_values, observed_time, observed_values,
                   time_buffer: timedelta,
                   wrapping_angle=None) -> Union[
    None, timedelta]:
    """

    :param model_time:
    :param model_values:
    :param observed_time:
    :param observed_values:
    :param time_buffer:
    :param wrapping_angle:
    :return:
    """
    if not isinstance(model_values, List):
        raise Exception('not a list')

    model = [[] for x in model_values]
    observed = [[] for x in model_values]
    time = None
    if wrapping_angle is None:
        wrapping_angle = [None for x in model_values]

    ii = -1
    for model_variable, observed_variable, wrap in zip(model_values,
                                                       observed_values,
                                                       wrapping_angle):
        out = get_overlapped_section(model_time, model_variable,
                                     observed_time, observed_variable,
                                     time_buffer=time_buffer,
                                     wrapping_angle=wrap)
        ii += 1
        if out:
            time, model[ii], observed[ii] = out
        else:
            return None

    time_delta = (time[2] - time[1])
    nbuffer = int(time_buffer / time_delta)
    out = numpy.zeros(2 * nbuffer + 1)
    for ii in range(-nbuffer, nbuffer + 1):
        for jj in range(len(model)):
            lagged_model = numpy.roll(model[jj], ii)

            if ii > 0:
                lagged_model[:ii] = numpy.nan
            elif ii < 0:
                lagged_model[ii:] = numpy.nan

            out[ii + nbuffer] += fit_quality(observed[jj],
                                             lagged_model,
                                             wrapping_angle[jj])


    index_extreme = numpy.argmax(out)
    time_lag = - time_delta * (index_extreme - nbuffer)

    return time_lag
