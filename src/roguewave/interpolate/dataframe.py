import pandas
from pandas import DataFrame
from roguewave.interpolate.general import interpolate_periodic
from roguewave.tools.time import to_datetime64
import numpy

def interpolate_dataframe_time(dataframe: DataFrame, new_time: numpy.ndarray) -> DataFrame:
    """
    A function to interpolate data in a dataframe. We need this function to be able to interpolate wrapped variables
    (e.g.longitudes and directions).
    """

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

        # Interpolation does not work on anything other than numeric types. Fixes a crash due to the new
        # "processing_source" adding a string to Spotter Api data that descibes where the data was processed.
        # We used to check for "object" to be more general - but that did not work when pandas switched to a new string
        # type. I reverted to just being specific on which column to ignore here. If we need a more general solution
        # later - let's implement it then.
        if name == 'processing_source':
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
