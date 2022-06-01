from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, \
    WaveSpectrum1DInput
import typing
from typing import Dict,List,Union, overload
from pysofar.spotter import Spotter, SofarApi
from datetime import datetime
from roguewave.tools import datetime_to_iso_time_string, to_datetime
from datetime import timedelta
from .exceptions import ExceptionNoFrequencyData
from pandas import read_csv, to_numeric
from roguewave.tools import _print
import numpy
import os
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

MAX_LOCAL_LIMIT = 20
MAXIMUM_NUMBER_OF_WORKERS = 10

def _get_spectrum_from_sofar_spotter_api(
        spotter: Spotter,
        start_date: typing.Union[datetime, str] = None,
        end_date: typing.Union[datetime, str] = None,
        limit: int = MAX_LOCAL_LIMIT,
) -> typing.List[WaveSpectrum1D]:
    """
    Grabs the requested spectra for this spotter based on the given keyword arguments

    :param limit: The limit for data to grab. Defaults to 20, For frequency data max of 100 samples at a time.
    :param start_date: ISO 8601 formatted date string. If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string. If not included defaults to end of spotter history

    :return: Data as a FrequencyDataList Object
    """


    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    json_data = spotter.grab_data(
        limit=limit,
        start_date=datetime_to_iso_time_string(start_date),
        end_date=datetime_to_iso_time_string(end_date),
        include_frequency_data=True,
        include_directional_moments=True)

    if not json_data['frequencyData']:
        raise ExceptionNoFrequencyData(
            f'Spotter {spotter.id} has no spectral data for the requested time range')

    out = []
    for spectrum in json_data['frequencyData']:
        # Load the data into the input object- this is a one-to-one
        # mapping of keys between dictionaries.
        wave_spectrum_input = WaveSpectrum1DInput(**spectrum)
        out.append(WaveSpectrum1D(wave_spectrum_input))

    return out

@overload
def get_spectrum_from_sofar_spotter_api(
        spotter_ids: List[str],
        start_date: Union[datetime, str] = None,
        end_date: Union[datetime, str] = None,
        session: SofarApi=None,
        verbose = False,
        limit=None
) -> Dict[str, List[WaveSpectrum1D]]: ...

@overload
def get_spectrum_from_sofar_spotter_api(
        spotter_ids: str,
        start_date: Union[datetime, str] = None,
        end_date: Union[datetime, str] = None,
        session: SofarApi=None,
        verbose = False,
        limit=None
) -> List[WaveSpectrum1D]: ...

def get_spectrum_from_sofar_spotter_api(
        spotter_ids: Union[str,List[str]],
        start_date: Union[datetime, str] = None,
        end_date: Union[datetime, str] = None,
        session: SofarApi=None,
        verbose = False,
        limit=None,
        parallel_download=True
) -> Union[ Dict[str, List[WaveSpectrum1D]], List[WaveSpectrum1D]]:
    """
    Grabs the requested spectra for this spotter based on the given keyword arguments

    :param limit: The limit for data to grab. Defaults to 20, For frequency data max of 100 samples at a time.
    :param start_date: ISO 8601 formatted date string. If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string. If not included defaults to end of spotter history

    :return: Data as a FrequencyDataList Object
    """


    if not isinstance( spotter_ids, list ):
        spotter_ids = [spotter_ids]
        return_list = False
    else:
        return_list = True

    if session is None:
        session = SofarApi()


    data = {}
    def worker( spotter_id):
        return _download_spectra(spotter_id,session,start_date,end_date,limit,verbose)

    if parallel_download:
        with ThreadPool(processes=min(cpu_count(),MAXIMUM_NUMBER_OF_WORKERS)) as pool:
            out = pool.map(worker, spotter_ids)

        for spotter_id, spectra in zip(spotter_ids,out):
            data[spotter_id] = spectra
    else:
        for spotter_id in spotter_ids:
            _print(verbose, f'Downloading data for spotter {spotter_id}')
            data[spotter_id] = _download_spectra(spotter_id,session,start_date,end_date,limit)

    if not return_list:
        return data[spotter_ids[0]]
    else:
        return data

def _download_spectra(spotter_id,session,start_date,end_date,limit,verbose):

    spotter = Spotter(spotter_id, spotter_id, session=session)
    _start_date = start_date
    data = []
    while True:
        # We can only download a maximum of 20 spectra at a time; so we need
        # to loop our request. We do not know how many spotters there are
        # in the given timeframe.
        #
        # assumptions:
        #   - spotter api returns a maximum of 20 items per requests
        #   - requests returned start from the requested start data and
        #     with the last entry either being the last entry that fits
        #     in the requested window, or merely the last sample that fits
        #     in the 20 items.
        #   - requests returned are in order

        if limit:
            # If we hit the limit (if given) of spotters requested, break
            if len(data) >= limit:
                break

            # otherwise, update our next request so we don't overrun
            # the limit
            local_limit = min(limit - len(data), MAX_LOCAL_LIMIT)
        else:
            # if no limit is given, just ask for the maximum allowed.
            local_limit = MAX_LOCAL_LIMIT

        try:
            # Try to get the next batch of spectra
            next = _get_spectrum_from_sofar_spotter_api(spotter, _start_date,
                                                        end_date, local_limit)

        except ExceptionNoFrequencyData as e:
            # If no frequency data was returned, we either...
            if not len(data):
                # ...raise an error, if no data was returned previously (no
                # data available at all)...
                raise e
            else:
                # ...or return, assuming that we exhausted the data was
                # available.
                break

        # Add the data to the list
        data += next

        # If we did not receive all data we requested...
        if len(next) < local_limit:
            # , we are done...
            break
        else:
            # ... else we update the startdate to be the timestamp of the last
            # known entry we received plus a second, and use this as the new
            # start.
            _start_date = to_datetime(next[-1].timestamp) + timedelta(
                seconds=1)
    return data


def get_spectrum_from_parser_output(path: str)->typing.List[WaveSpectrum1D]:
    """

    :param path: Path that contains the output from the spotter parser.
    :return: A list of WaveSpectrum1D objects.
    """
    def load_spectral_file(file):
        if os.path.isfile(file):
            data = read_csv(file).apply(to_numeric, errors='coerce')
        else:
            raise FileNotFoundError(file)

        columns = list(data.columns)
        frequencies = [float(x) for x in columns[8:]]
        values = data[columns[8:]].values

        time_tuple = data[columns[0:6]].values
        time = []
        for index in range(time_tuple.shape[0]):
            time.append(
                datetime(year=time_tuple[index, 0], month=time_tuple[index, 1],
                         day=time_tuple[index, 2], hour=time_tuple[index, 3],
                         minute=time_tuple[index, 4],
                         second=time_tuple[index, 5]))

        return {'time': numpy.array(time), 'frequencies': frequencies,
                'values': values}

    files = ['a1', 'b1', 'a2', 'b2', 'Szz']
    data = {}
    for file_type in files:
        file_location = os.path.join(path, file_type+'.csv')
        data[file_type] = load_spectral_file(file_location)

    number_of_spectra = data['Szz']['values'].shape[0]
    spectra = []
    for index in range(0,number_of_spectra):
        input = WaveSpectrum1DInput(frequency=data['Szz']['frequencies'],
                                    varianceDensity=data['Szz']['values'][index,:],
                                    timestamp=data['Szz']['time'][index], latitude=None,
                                    longitude=None, a1=data['a1']['values'][index,:],
                                    b1=data['b1']['values'][index,:],
                                    a2=data['a2']['values'][index,:],
                                    b2=data['b2']['values'][index,:])
        spectra.append(WaveSpectrum1D(input))
    return spectra
