from pysofar.sofar import SofarApi
from typing import List, Dict, MutableMapping, Union, Any
from pandas import DataFrame
from numpy import unique, inf, array
from roguewave import (
    FrequencySpectrum,
    create_1d_spectrum,
    to_datetime64,
    to_datetime_utc,
)
from pandas import Timestamp

# SofarAPI instance.
_API = None


# Helper functions
def get_spotter_ids(sofar_api: SofarApi = None) -> List[str]:
    """
    Get a list of Spotter ID's that are available through this account.

    :param sofar_api: valid SofarApi instance.
    :return: List of spotters available through this account.
    """
    if sofar_api is None:
        sofar_api = _get_sofar_api()
    return sofar_api.device_ids


# -----------------------------------------------------------------------------


def _get_sofar_api() -> SofarApi:
    """
    Gets a new sofar API object if requested. Returned object is essentially a
    Singleton class-> next calls will return the stored object instead of
    creating a new class. For module internal use only.

    :return: instantiated SofarApi object
    """
    global _API
    if _API is None:
        _API = SofarApi()
    return _API


# -----------------------------------------------------------------------------


# 6) Helper Functions
# =============================================================================
def _unique_filter(data):
    """
    Filter for dual time entries that occur due to bugs in wavefleet (same
    record returned twice)
    :param data:
    :return:
    """
    if len(data) == 0:
        return data

    # Get time
    if isinstance(data[0], FrequencySpectrum):
        time = array([to_datetime_utc(x.time.values).timestamp() for x in data])
    else:
        time = array([x["time"].timestamp() for x in data])

    # Get indices of unique times
    _, unique_indices = unique(time, return_index=True)

    # Return only unique indices
    return [data[index] for index in unique_indices]


# -----------------------------------------------------------------------------


def _none_filter(data: Dict):
    """
    Filter for the occasional occurance of bad data returned from wavefleet.
    :param data:
    :return:
    """
    return list(
        filter(
            lambda x: (x["latitude"] is not None)
            and (x["longitude"] is not None)
            and (x["timestamp"] is not None),
            data,
        )
    )


# -----------------------------------------------------------------------------


def _get_class(key, data) -> Union[MutableMapping, FrequencySpectrum]:
    MAP_API_TO_MODEL_NAMES = {
        "waves": {"timestamp": "time"},
        "surfaceTemp": {
            "timestamp": "time",
            "degrees": "seaSurfaceTemperature",
        },
        "wind": {
            "timestamp": "time",
            "speed": "windVelocity10Meter",
            "direction": "windDirection10Meter",
        },
        "barometerData": {"timestamp": "time", "value": "seaSurfacePressure"},
        "microphoneData": {"timestamp": "time", "value": "soundPressure"},
        "smartMooringData": {
            "timestamp": "time",
        },
    }

    if key == "frequencyData":
        return create_1d_spectrum(
            frequency=array(data["frequency"]),
            variance_density=array(data["varianceDensity"]),
            time=to_datetime64(data["timestamp"]),
            latitude=array(data["latitude"]),
            longitude=array(data["longitude"]),
            a1=array(data["a1"]),
            b1=array(data["b1"]),
            a2=array(data["a2"]),
            b2=array(data["b2"]),
            dims=("frequency",),
            depth=inf,
        )
    else:
        out = {}
        # Here we postprocess non-spectral data. We do three things:
        # 1) rename variable names to standard names
        # 2) apply any postprocessing if appropriate (e.g. convert datestrings to datetimes)
        # 3) add any desired postprocess variables- this is legacy for peak frequency. We
        #    should not add any variables here anymore.
        if key in MAP_API_TO_MODEL_NAMES:
            # Get the mapping from API variable names to standard variable names.
            mapping = MAP_API_TO_MODEL_NAMES[key]
            for var_key, value in data.items():
                # If the name differs get the correct name, otherwise use the
                # name returned from the API
                target_key = mapping[var_key] if var_key in mapping else var_key

                if target_key == "time":
                    out[target_key] = to_datetime_utc(value)
                else:
                    out[target_key] = value

            if key == "waves":
                out["peakFrequency"] = 1 / out["peakPeriod"]

            return out

        else:
            raise Exception(f"Unknown variable {key}")


def as_dataframe(list_of_dict: List[Dict[str, Any]]) -> DataFrame:
    """
    Convert a list of dictionaries to a dataframe. Each dictionary in the list is assumed
    to have the same set of keys.

    :param list_of_dict:
    :return: dataframe with
    """
    data = {key: [] for key in list_of_dict[0].keys()}
    for dictionary in list_of_dict:
        for key in data:
            data[key].append(dictionary[key])
    data["time"] = [Timestamp(x) for x in data["time"]]
    dataframe = DataFrame.from_dict(data)
    # dataframe.set_index("time", inplace=True)

    return dataframe


# -----------------------------------------------------------------------------
