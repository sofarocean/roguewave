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
)
from pandas import DataFrame
from typing import Union, Dict, List
from datetime import datetime
import gzip
import json
import numpy
import base64
from io import BytesIO
from roguewave import filecache
from xarray import Dataset, open_dataset, DataArray
import boto3
from botocore.errorfactory import ClientError
import tempfile
import pickle
import os
from roguewave.tools.time import to_datetime64


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


class DataEncoder(json.JSONEncoder):
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
            if 'time' in obj.columns:
                obj['time'] = to_datetime64(obj['time'])
            return _b64_encode_dataset(Dataset.from_dataframe(obj), "DataFrame")

        else:
            return json.JSONEncoder.default(self, obj)


def object_hook(dictionary: dict):
    if "__class__" in dictionary:
        if dictionary["__class__"] == "DataFrame":
            return _b64_decode_dataset(dictionary["data"]).to_dataframe()
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


def load(filename: str, force_redownload_if_remote=False, filetype="roguewave"):
    """
    Load spectral data as saved by "save_spectrum" from the given file and
    return a (nested) object. The precise format of the output depends on what
    was saved.

    :param filename: path to file to load. If an s3 uri is given (of the form s3://bucket/key) the remote file is
        downloaded and cached locally. Future requests of the same uri are retrieved from the cache.

    :param force_redownload_if_remote: If s3 file is cached force a refresh with the remote resource.

    :param filetype: ['roguewave', 'pickle', 'netcdf']. Type of file to be loaded.

    :return:
        - Data in the same form it was saved.
    """

    if filename.startswith("s3://"):
        # Get the file from s3 and cache locally
        if force_redownload_if_remote:
            filecache.delete_files(filename, error_if_not_in_cache=False)
        filename = filecache.filepaths([filename])[0]

    if filetype == "roguewave":
        with gzip.open(filename, "rb") as file:
            data = file.read().decode("utf-8")
        return json.loads(data, object_hook=object_hook)

    elif filetype == "pickle":
        with open(filename, "rb") as file:
            data = pickle.load(file)
        return data

    elif filetype == "netcdf":
        return open_dataset(filename)


def save(
    _input: _UNION, filename: str, overwrite=True, s3_overwrite=False, use_pickle=False
):
    """
    Save roguewave data in JSON form compressed with gzip.

    :param _input:
        - Data containing python primitives (dict/list) or any of the RogueWave
          classes.

    :param filename: path to save data. If an s3 uri is given (of the form s3://bucket/key) the  file is
        saved to s3. If a local file already exists with the same name the file is overwritten if overwrite = True
        (default), otherwise an error is raised. If an s3 object with the same uri exists we raise an error (unless
        s3_overwrite = True).

    :param overwrite: Default True. If False an error is raised if a file with the same name already exists. By default
        we simply overwrite under the assumption we are aware of local context.

    :param s3_overwrite: Default False. If False an error is raised if an object with the same uri already exists on s3.
        By default we raise an error as we may clash with keys from others.

    :param use_pickle: Default False. Use the pickle protocol to save the object

    :return: None
    """

    def write(filename, _input):
        if use_pickle:
            with open(filename, "wb") as file:
                pickle.dump(_input, file)
        else:
            with gzip.open(filename, "wt") as file:
                json.dump(_input, file, cls=DataEncoder)

    if filename.startswith("s3://"):
        with tempfile.TemporaryFile() as temp_file:
            write(temp_file, _input)
            temp_file.seek(0)

            bucket, key = filename.removeprefix("s3://").split("/", 1)
            s3 = boto3.client("s3")

            try:
                s3.head_object(Bucket=bucket, Key=key)
                if not s3_overwrite:
                    raise FileExistsError(
                        f"Key {key} already exists in bucket {bucket}. To overwrite the file (if"
                        f"desired) set s3_overwrite=True"
                    )
            except ClientError:
                # Not found
                pass

            s3.upload_fileobj(temp_file, bucket, key)

    else:
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"Key {filename} already exists. To overwrite the file (if"
                f"desired) set overwrite=True"
            )
        write(filename, _input)
