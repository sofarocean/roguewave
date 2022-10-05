from roguewave.wavespectra import (
    FrequencySpectrum,
    create_1d_spectrum,
    concatenate_spectra,
)
from datetime import datetime
from pandas import read_csv, to_numeric
import numpy
import os


def get_spectrum_from_parser_output(path: str) -> FrequencySpectrum:
    """

    :param path: Path that contains the output from the spotter parser.
    :return: A list of WaveSpectrum1D objects.
    """

    def load_spectral_file(file):
        if os.path.isfile(file):
            data = read_csv(file).apply(to_numeric, errors="coerce")
        else:
            raise FileNotFoundError(file)

        columns = list(data.columns)
        frequencies = [float(x) for x in columns[8:]]
        values = data[columns[8:]].values

        time_tuple = data[columns[0:6]].values
        time = []
        for index in range(time_tuple.shape[0]):
            time.append(
                datetime(
                    year=time_tuple[index, 0],
                    month=time_tuple[index, 1],
                    day=time_tuple[index, 2],
                    hour=time_tuple[index, 3],
                    minute=time_tuple[index, 4],
                    second=time_tuple[index, 5],
                )
            )

        return {"time": numpy.array(time), "frequencies": frequencies, "values": values}

    files = ["a1", "b1", "a2", "b2", "Szz"]
    data = {}
    for file_type in files:
        file_location = os.path.join(path, file_type + ".csv")
        data[file_type] = load_spectral_file(file_location)

    number_of_spectra = data["Szz"]["values"].shape[0]
    spectra = []
    for index in range(0, number_of_spectra):
        spectra.append(
            create_1d_spectrum(
                frequency=data["Szz"]["frequencies"],
                variance_density=data["Szz"]["values"][index, :],
                time=data["Szz"]["time"][index],
                latitude=None,
                longitude=None,
                a1=data["a1"]["values"][index, :],
                b1=data["b1"]["values"][index, :],
                a2=data["a2"]["values"][index, :],
                b2=data["b2"]["values"][index, :],
            )
        )
    return concatenate_spectra(spectra, dim="time")
