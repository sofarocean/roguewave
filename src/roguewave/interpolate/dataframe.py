from pandas import DataFrame
from roguewave.interpolate.general import interpolate_periodic
from roguewave.tools.time import to_datetime64, to_datetime_utc
import numpy

def interpolate_dataframe_time( dataframe:DataFrame, new_time:numpy.ndarray):

    output = DataFrame(index=to_datetime_utc(new_time))
    columns = list(dataframe.columns)
    old_time = to_datetime64(dataframe.index.values)
    new_time = to_datetime64(new_time)


    # Todo more general direction detection.

    for name in columns:
        name: str
        period = None
        if 'direction' in name.lower():
            fp_discont = 360
            fp_period  = 360
        else:
            fp_discont = None
            fp_period  = None

        output[name] = interpolate_periodic(
            old_time.astype('float64'),
            dataframe[name].values,
            new_time.astype('float64'),
            x_period=period,
            fp_period=fp_period,
            fp_discont=fp_discont
        )
    return output