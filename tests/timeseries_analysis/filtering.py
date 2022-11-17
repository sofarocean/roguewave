from roguewave.spotter.read_csv_data import read_gps, read_displacement, read_spectra
from roguewave.timeseries_analysis.pipeline import pipeline
from pandas import DataFrame
from numpy.typing import NDArray
from roguewave import FrequencySpectrum


def get_displacement() -> DataFrame:
    return read_displacement(path="/Users/pietersmit/Downloads/Sunflower13/log")


def get_gps() -> DataFrame:
    return read_gps(path="/Users/pietersmit/Downloads/Sunflower13/log")


def get_spectra() -> FrequencySpectrum:
    return read_spectra(path="/Users/pietersmit/Downloads/Sunflower13/log")


def integrate_gps(gps: DataFrame) -> NDArray:
    w = gps["w"].values
    time = gps["time"].values
    z = pipeline(time, w)
    return z


if __name__ == "__main__":
    spec = get_spectra()

    import matplotlib.pyplot as plt

    plt.plot(spec.time, spec.significant_waveheight)
    plt.show()
    print("ja")
    # gps = get_gps()
    # z = integrate_gps(gps)
    # disp = get_displacement()
    # import matplotlib.pyplot as plt
    #
    # # plt.plot( disp['time'].values[1000:2000], disp['z'].values[1000:2000]  )
    # # plt.plot(gps['time'].values[1000:2000], z[1000:2000],'k')
    # plt.plot(disp["time"].values, disp["z"].values)
    # plt.plot(gps["time"].values, z, "rx")
    # # plt.plot(gps["time"].values, gps["z"], "r")
    # plt.xlim((1.663445e9, 1.6634453e9))
    # plt.ylim((-5, 5))
    #
    # plt.figure()
    # plt.plot(gps["time"], gps["w"], "k")
    # plt.plot(
    #     gps["time"],
    #     exponential_filter(gps["time"].values, gps["w"].values),
    #     "r",
    # )
    #
    # plt.xlim((1.663445e9, 1.6634451e9))
    #
    # plt.show()
