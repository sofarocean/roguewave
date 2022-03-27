from roguewave.wavespectra.spectrum1D import WaveSpectrum1D, \
    WaveSpectrum1DInput
import typing
from pysofar.spotter import Spotter
from datetime import datetime
from roguewave.tools import datetime_to_iso_time_string
from .exceptions import ExceptionNoFrequencyData

def get_spectrum_from_sofar_spotter_api(
        spotter: Spotter,
        start_date: typing.Union[datetime, str] = None,
        end_date: typing.Union[datetime, str] = None,
        limit: int = 20,
) -> typing.List[WaveSpectrum1D]:
    """
    Grabs the requested spectra for this spotter based on the given keyword arguments

    :param limit: The limit for data to grab. Defaults to 20, For frequency data max of 100 samples at a time.
    :param start_date: ISO 8601 formatted date string. If not included defaults to beginning of spotters history
    :param end_date: ISO 8601 formatted date string. If not included defaults to end of spotter history

    :return: Data as a FrequencyDataList Object
    """

    start_date = datetime_to_iso_time_string(start_date)
    end_date = datetime_to_iso_time_string(end_date)

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
        # mapping of keys between dictionaries (we would not _need_ to
        # map onto a input object-> we could pass spectrum directly to
        # the constructor. But it is clearer to fail here if something
        # is amiss with the api JSON response format)
        wave_spectrum_input = WaveSpectrum1DInput(**spectrum)

        out.append(WaveSpectrum1D(wave_spectrum_input))

    return out
