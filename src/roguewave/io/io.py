"""
Contents: IO

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines that can be used to save and load data.

Functions:

How To Use This Module
======================
(See the individual functions for details.)

"""
from roguewave.wavespectra import (
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
    create_1d_spectrum,
)
from pandas import DataFrame, read_json
from typing import Union, Dict, List
from datetime import datetime
import gzip
import json
import os
import numpy
import base64
from io import BytesIO
from roguewave import to_datetime_utc
from xarray import Dataset, open_dataset, DataArray


_UNION = Union[
    FrequencySpectrum,
    FrequencyDirectionSpectrum,
    List[FrequencySpectrum],
    Dict[str, List[FrequencyDirectionSpectrum]],
    Dict[str, List[FrequencySpectrum]],
    List[List[FrequencyDirectionSpectrum]],
    List[List[DataFrame]],
    Dict[int, List[DataFrame]],
    Dict[str, DataFrame],
]


def _b64_encode_numpy(val: numpy.ndarray) -> dict:
    if val.dtype == numpy.float64:
        val = val.astype(dtype=numpy.float32)

    if val.dtype == numpy.int64:
        val = val.astype(dtype=numpy.int32)

    dtype = val.dtype.str
    shape = val.shape
    data = base64.b64encode(val)

    return {
        "__class__": "ndarray",
        "data": {"dtype": dtype, "shape": shape, "data": data.decode("utf8")},
    }


def _b64_decode_numpy(data):
    decoded_bytes = base64.b64decode(bytes(data["data"], encoding="utf-8"))
    dtype = numpy.dtype(data["dtype"])
    array = numpy.frombuffer(decoded_bytes, dtype=dtype)
    return numpy.reshape(array, data["shape"])


def _b64_encode_dataset(val: Dataset, name) -> dict:
    net_cdf = val.to_netcdf(engine="scipy")
    data = base64.b64encode(net_cdf)
    return {"__class__": name, "data": {"data": data.decode("utf8")}}


def _b64_encode_dataarray(val: DataArray, name) -> dict:
    dataset = Dataset()
    dataset = dataset.assign({"dataarray": val})
    net_cdf = dataset.to_netcdf(engine="scipy")
    data = base64.b64encode(net_cdf)
    return {"__class__": name, "data": {"data": data.decode("utf8")}}


def _b64_decode_dataset(data) -> Dataset:
    decoded_bytes = base64.b64decode(bytes(data["data"], encoding="utf-8"))
    with BytesIO(decoded_bytes) as fp:
        dataset = open_dataset(fp, engine="scipy")
    return dataset


def _b64_decode_dataarray(data) -> DataArray:
    decoded_bytes = base64.b64decode(bytes(data["data"], encoding="utf-8"))
    with BytesIO(decoded_bytes) as fp:
        dataset = open_dataset(fp, engine="scipy")
    return dataset["dataarray"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return _b64_encode_numpy(obj)
        elif isinstance(obj, numpy.int32):
            return int(obj)
        elif isinstance(obj, numpy.float32):
            return float(obj)
        elif isinstance(obj, numpy.int64):
            return int(obj)
        elif isinstance(obj, numpy.float64):
            return float(obj)
        elif isinstance(obj, Dataset):
            return _b64_encode_dataset(obj, "Dataset")
        elif isinstance(obj, DataArray):
            return _b64_encode_dataarray(obj, "DataArray")
        elif isinstance(obj, FrequencySpectrum):
            return _b64_encode_dataset(obj.dataset, "FrequencySpectrum")
        elif isinstance(obj, FrequencyDirectionSpectrum):
            return _b64_encode_dataset(obj.dataset, "FrequencyDirectionSpectrum")
        elif isinstance(obj, datetime):
            return {"__class__": "datetime", "data": datetime.isoformat(obj)}
        elif isinstance(obj, DataFrame):
            return {"__class__": "DataFrame", "data": obj.to_json(date_unit="s")}
        else:
            return json.JSONEncoder.default(self, obj)


def object_hook(dictionary: dict):
    if "__class__" in dictionary:
        if dictionary["__class__"] == "DataFrame":
            #
            df = read_json(dictionary["data"])
            if "timestamp" in df:
                df["timestamp"] = df["timestamp"].apply(lambda x: x.tz_localize("utc"))
            elif "time" in df:
                time = to_datetime_utc(df["time"].values)
                # df["time"] = df["time"].apply(lambda x: x.tz_localize("utc"))
                df["time"] = time
            elif "valid_time" in df:
                df["valid_time"] = df["valid_time"].apply(
                    lambda x: x.tz_localize("utc")
                )
            elif "init_time" in df:
                df["valid_time"] = df["valid_time"].apply(
                    lambda x: x.tz_localize("utc")
                )
            else:
                df.index = [x.tz_localize("utc") for x in df.index]
            return df
        elif dictionary["__class__"] == "WaveSpectrum1D":
            data = dictionary["data"]
            return create_1d_spectrum(
                frequency=data["frequency"],
                variance_density=data["varianceDensity"],
                time=data["timestamp"],
                latitude=data["latitude"],
                longitude=data["longitude"],
                a1=data["a1"],
                b1=data["b1"],
                a2=data["a2"],
                b2=data["b2"],
            )
        elif dictionary["__class__"] == "Dataset":
            return _b64_decode_dataset(dictionary["data"])
        elif dictionary["__class__"] == "DataArray":
            return _b64_decode_dataarray(dictionary["data"])
        elif dictionary["__class__"] == "FrequencySpectrum":
            return FrequencySpectrum(_b64_decode_dataset(dictionary["data"]))
        elif dictionary["__class__"] == "FrequencyDirectionSpectrum":
            return FrequencyDirectionSpectrum(_b64_decode_dataset(dictionary["data"]))
        elif dictionary["__class__"] == "ndarray":
            return _b64_decode_numpy(dictionary["data"])
        elif dictionary["__class__"] == "datetime":
            return datetime.fromisoformat(dictionary["data"])
    else:
        return dictionary


def load(filename: str) -> _UNION:
    """
    Load spectral data as saved by "save_spectrum" from the given file and
    return a (nested) object. The precise format of the output depends on what
    was saved.

    :param filename: path to file to load.
    :return:
        - Data in the same form it was saved.
    """

    with gzip.open(filename, "rb") as file:
        data = file.read().decode("utf-8")
    return json.loads(data, object_hook=object_hook)


def save(
    _input: _UNION, filename: str, format="json", separate_spotters_into_files=False
):
    """
    Save spectral data, possible in nested form as returned by the spectral
    partition/reconstruction/etc. functions. Data is saved in JSON form
    compressed with gzip.

    :param _input:
        - Data containing python primitives (dict/list) or any of the RogueWave
          classes.

    :param filename: path to save data.
    :return: None
    """

    def write(filename, _input, format):
        if format == "json":
            with gzip.open(filename, "wt") as file:
                json.dump(_input, file, cls=NumpyEncoder)
                # file.write(data.encode('utf-8'))
        else:
            raise Exception("Unknown output format")

    if separate_spotters_into_files:
        os.makedirs(filename, exist_ok=True)
        for key in _input:
            name = os.path.join(filename, key)
            write(name, _input[key], format)

    else:
        write(filename, _input, format)
