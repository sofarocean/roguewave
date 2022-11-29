from pandas import DataFrame
from roguewave.interpolate.general import interpolate_periodic
from roguewave.tools.time import to_datetime64
import numpy


def interpolate_dataframe_time(dataframe: DataFrame, new_time: numpy.ndarray):

    output = DataFrame()
    output["time"] = new_time
    columns = list(dataframe.columns)
    old_time = to_datetime64(dataframe["time"].values)
    new_time = to_datetime64(new_time)

    for name in columns:
        name: str
        period = None
        if "direction" in name.lower():
            fp_discont = 360
            fp_period = 360
        else:
            fp_discont = None
            fp_period = None

        if name == "time":
            continue

        # Interpolation does not work on anything other than numeric types. If we have an object - we just ignore it
        # (do not include column in output). Fixes a crash due to the new "processing_source" adding a string to
        # Spotter Api data that descibes where the data was processed.
        if dataframe[name].dtype == numpy.dtype(object):
            print(name)
            continue

        output[name] = interpolate_periodic(
            old_time.astype("float64"),
            dataframe[name].values,
            new_time.astype("float64"),
            x_period=period,
            fp_period=fp_period,
            fp_discont=fp_discont,
        )
    return output
