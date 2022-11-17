from roguewave import FrequencySpectrum
from roguewave.timeseries_analysis.welch import estimate_spectrum
from roguewave.spotter.read_csv_data import read_displacement, read_spectra
from pandas import DataFrame


def get_displacement() -> DataFrame:
    return read_displacement(path="/Users/pietersmit/Downloads/Sunflower13/log")


def calc_spectra() -> FrequencySpectrum:
    disp = get_displacement()
    spec = estimate_spectrum(
        disp["time"].values, disp["x"].values, disp["y"].values, disp["z"].values
    )
    return spec


def get_spectra() -> FrequencySpectrum:
    return read_spectra(path="/Users/pietersmit/Downloads/Sunflower13/log")


if __name__ == "__main__":

    d = calc_spectra()
    specta = get_spectra()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(d.time, d.significant_waveheight, "k")
    plt.plot(specta.time, specta.significant_waveheight, "r")
    plt.show()

    pass
