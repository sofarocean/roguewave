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
from roguewave.wavespectra.spectrum2D import WaveSpectrum2D, \
    WaveSpectrum2DInput
import typing
import netCDF4
import numpy


def get_token():
    # config values
    userpath = os.path.expanduser("~")
    enviromentFile = os.path.join(userpath, 'sofar_api.env')
    dotenv.load_dotenv(enviromentFile)
    token = os.getenv('SPECTRAL_API_TOKEN')
    _wavefleet_token = token

    return _wavefleet_token


def get_endpoint():
    _endpoint = os.getenv('SPECTRAL_URL')
    if _endpoint is None:
        _endpoint = 'https://api.sofarocean.com/api'
    return _endpoint


class SofarSpectralAPI(SofarConnection):
    def __init__(self, costum_token=None):
        #
        token = costum_token if costum_token else get_token()
        super().__init__(token)
        self.endpoint = get_endpoint()

    @cached_property
    def _query(self):
        scode, data = self._get('op-wave-spectra')

        if scode != 200:
            raise QueryError(data['message'])

        return data['data']

    def _download(self, latitude, longitude):
        # Helper methods to download files

        url = f"{self.endpoint}/op-wave-spectra/{latitude}/{longitude}"
        response = requests.get(url, headers=self.header)

        status = response.status_code
        if status != 200:
            data = json.loads(response.text)
            raise QueryError(data['message'])

        data = response.content
        return data

    def points(self):
        data = self._query
        print(data)
        return [{'latitude': x['latitude'], 'longitude': x['longitude']} for x
                in data]

    def urls(self):
        data = self._query
        return [x['url'] for x in data]

    def download_spectral_file(self, latitude, longitude, directory) -> str:
        os.makedirs(directory, exist_ok=True)
        filename = f'spectrum_{longitude}E_{latitude}N.nc'
        filepath = os.path.join(directory, filename)

        with open(filepath, 'wb') as file:
            file.write(self._download(latitude, longitude))

        return filepath

    def download_all_spectral_files(self, directory) -> typing.List[str]:
        points = self.points()
        return [
            self.download_spectral_file(point['latitude'], point['longitude'],
                                        directory) for point in points]

def load_spectral_file(file)->typing.List[WaveSpectrum2D]:
    dataset = netCDF4.Dataset(file)

    nt = dataset.dimensions['time'].size
    frequencies = dataset.variables['frequencies'][:]
    directions = dataset.variables['directions'][:]

    spectra = []
    for ii in range(0,nt):
        unixepochtime = dataset.variables['time'][ii]
        latitude = dataset.variables['latitude'][0]
        longitude = dataset.variables['longitude'][0]
        variance_density = \
            numpy.squeeze(dataset.variables['frequency_direction_spectrum'][ii,:,:])

        spectrum2D_input = WaveSpectrum2DInput(
            frequency=frequencies,varianceDensity=variance_density,
            timestamp=unixepochtime,latitude=latitude,longitude=longitude,
            directions=directions
        )
        spectra.append(WaveSpectrum2D(spectrum2D_input))
    return spectra