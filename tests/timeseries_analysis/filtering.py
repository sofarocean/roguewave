from roguewave.timeseries_analysis.parse_spotter_files import (
    load_displacement,
    load_gps,
)
from roguewave.timeseries_analysis.filtering import (
    integrate,
    outlier_rejection_filter,
    exponential_highpass_filter,
)
from pandas import DataFrame
from numpy.typing import NDArray


def get_displacement() -> DataFrame:
    return load_displacement(path="/Users/pietersmit/Downloads/Sunflower13/log")


def get_gps() -> DataFrame:
    return load_gps(path="/Users/pietersmit/Downloads/Sunflower13/log")


def integrate_gps(gps: DataFrame) -> NDArray:
    w = gps["w"].values
    time = gps["time"].values
    w = outlier_rejection_filter(time, w)
    w = exponential_highpass_filter(time, w)
    z = integrate(time, w)
    z = exponential_highpass_filter(time, z)
    return z


if __name__ == "__main__":
    gps = get_gps()
    z = integrate_gps(gps)
    disp = get_displacement()
    import matplotlib.pyplot as plt

    # plt.plot( disp['time'].values[1000:2000], disp['z'].values[1000:2000]  )
    # plt.plot(gps['time'].values[1000:2000], z[1000:2000],'k')
    plt.plot(disp["time"].values, disp["z"].values)
    plt.plot(gps["time"].values, z, "k")
    plt.plot(gps["time"].values, gps["z"], "r")
    plt.xlim((1.663445e9, 1.6634451e9))
    plt.ylim((-5, 5))

    plt.figure()
    plt.plot(gps["time"], gps["w"], "k")
    plt.plot(
        gps["time"],
        exponential_highpass_filter(gps["time"].values, gps["w"].values),
        "r",
    )

    wfilt = outlier_rejection_filter(gps["time"].values, gps["w"].values)
    plt.plot(
        gps["time"],
        exponential_highpass_filter(gps["time"].values, gps["w"].values),
        "r",
    )
    plt.plot(gps["time"], exponential_highpass_filter(gps["time"].values, wfilt), "b")
    plt.xlim((1.663445e9, 1.6634451e9))

    plt.show()
