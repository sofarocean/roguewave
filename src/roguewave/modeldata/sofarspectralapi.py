"""

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Smit
"""
import os
import dotenv
import requests
import json
from pysofar import SofarConnection
from pysofar.wavefleet_exceptions import QueryError
from functools import cached_property
from roguewave.wavespectra import (
    FrequencyDirectionSpectrum,
    create_2d_spectrum,
    concatenate_spectra,
)
import typing
import netCDF4
import numpy
from typing import List, overload, Union, Dict
from roguewave.tools.time import to_datetime64


def get_token():
    # config values
    userpath = os.path.expanduser("~")
    enviromentFile = os.path.join(userpath, "sofar_api.env")
    dotenv.load_dotenv(enviromentFile)
    token = os.getenv("SPECTRAL_API_TOKEN")
    _wavefleet_token = token

    return _wavefleet_token


def get_endpoint():
    _endpoint = os.getenv("SPECTRAL_URL")
    if _endpoint is None:
        _endpoint = "https://api.sofarocean.com/api"
    return _endpoint


class SofarSpectralAPI(SofarConnection):
    def __init__(self, costum_token=None):
        #
        token = costum_token if costum_token else get_token()
        super().__init__(token)
        self.endpoint = get_endpoint()

    @cached_property
    def _query(self):
        scode, data = self._get("op-wave-spectra")

        if scode != 200:
            raise QueryError(data["message"])

        return data["data"]

    def _download(self, latitude, longitude):
        # Helper methods to download files

        url = f"{self.endpoint}/op-wave-spectra/{latitude}/{longitude}"
        response = requests.get(url, headers=self.header)

        status = response.status_code
        if status != 200:
            data = json.loads(response.text)
            raise QueryError(data["message"])

        data = response.content
        return data

    def points(self):
        data = self._query
        return [{"latitude": x["latitude"], "longitude": x["longitude"]} for x in data]

    def urls(self):
        data = self._query
        return [x["url"] for x in data]

    def download_spectral_file(self, latitude, longitude, directory) -> str:
        os.makedirs(directory, exist_ok=True)
        filename = f"spectrum_{longitude}E_{latitude}N.nc"
        filepath = os.path.join(directory, filename)

        with open(filepath, "wb") as file:
            file.write(self._download(latitude, longitude))

        return filepath

    def download_all_spectral_files(self, directory) -> typing.List[str]:
        points = self.points()
        return [
            self.download_spectral_file(
                point["latitude"], point["longitude"], directory
            )
            for point in points
        ]


@overload
def load_sofar_spectral_file(filename: str) -> FrequencyDirectionSpectrum:
    ...


@overload
def load_sofar_spectral_file(filename: List[str]) -> List[FrequencyDirectionSpectrum]:
    ...


@overload
def load_sofar_spectral_file(
    filename: Dict[str, str]
) -> Dict[str, FrequencyDirectionSpectrum]:
    ...


def load_sofar_spectral_file(filename: Union[str, List[str], Dict[str, str]]):

    if isinstance(filename, str):
        if not os.path.exists(filename):
            raise FileNotFoundError()

        dataset = netCDF4.Dataset(filename)

        nt = dataset.dimensions["time"].size
        frequencies = dataset.variables["frequencies"][:]
        directions = dataset.variables["directions"][:]

        spectra = []
        for ii in range(0, nt):
            unixepochtime = dataset.variables["time"][ii]
            latitude = dataset.variables["latitude"][0]
            longitude = dataset.variables["longitude"][0]
            variance_density = numpy.squeeze(
                dataset.variables["frequency_direction_spectrum"][ii, :, :]
            )

            if (
                dataset.variables["frequency_direction_spectrum"].dimensions[1]
                == "directions"
            ):
                variance_density = variance_density.transpose()

            spectra.append(
                create_2d_spectrum(
                    frequency=frequencies,
                    variance_density=variance_density,
                    time=to_datetime64(unixepochtime),
                    latitude=latitude,
                    longitude=longitude,
                    direction=directions,
                )
            )
        return concatenate_spectra(spectra, dim="time")
    elif isinstance(filename, dict):
        out = {}
        for key in filename:
            out[key] = load_sofar_spectral_file(filename[key])
        return out
    elif isinstance(filename, List):
        return [load_sofar_spectral_file(name) for name in filename]
    else:
        raise Exception("Unsupported input type for filename")
