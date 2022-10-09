from tests.test_data.spectra import get_1d_spec
from roguewave.wavephysics.windSpotter import equilibrium_range_values, U10


def test_equilibrium_range_values():
    spec = get_1d_spec()
    e, a1, b1 = equilibrium_range_values(spec, method="peak")
    e, a1, b1 = equilibrium_range_values(spec, method="mean")


def test_U10():
    spec = get_1d_spec()
    u10a, _, _ = U10(spec, "peak")
    u10b, _, _ = U10(spec, "mean")

    import matplotlib.pyplot as plt

    plt.plot(u10a, "k0")
    plt.plot(u10b, "rx")
    plt.show()


if __name__ == "__main__":
    test_equilibrium_range_values()
