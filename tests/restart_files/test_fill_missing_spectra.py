import numpy
from xarray import Dataset
from roguewavespectrum import Spectrum
from roguewave.tools.grid import midpoint_rule_step
from roguewave.wavewatch3.restart_file import RestartFile


class _FakeRestartFile:
    """
    A minimal duck-typed stand-in for RestartFile, exposing only what
    fill_missing_spectra/_fill_spectrum actually use, so these can be
    tested without constructing a real, binary-file-backed RestartFile.
    """

    fill_missing_spectra = RestartFile.fill_missing_spectra
    _fill_spectrum = RestartFile._fill_spectrum

    def __init__(self, frequency, direction):
        self.frequency = frequency
        self.direction = direction

    @property
    def number_of_frequencies(self):
        return len(self.frequency)

    @property
    def number_of_directions(self):
        return len(self.direction)


def _make_spectrum(frequency, direction, values):
    number_of_points = values.shape[0]
    return Spectrum(
        Dataset(
            data_vars={
                "directional_variance_density": (
                    ("points", "frequency", "direction"),
                    values,
                ),
                "longitude": (("points",), numpy.zeros(number_of_points)),
                "latitude": (("points",), numpy.zeros(number_of_points)),
                "depth": (("points",), numpy.full(number_of_points, 100.0)),
            },
            coords={"frequency": frequency, "direction": direction},
        )
    )


def test_fill_spectrum_calm_is_zero():
    fake_restart_file = _FakeRestartFile(
        numpy.array([0.1, 0.2]), numpy.array([0.0, 90.0, 180.0, 270.0])
    )
    fill = fake_restart_file._fill_spectrum("calm")

    assert fill.shape == (2, 4)
    assert numpy.all(fill == 0.0)


def test_fill_spectrum_user_defined_passes_through():
    fake_restart_file = _FakeRestartFile(
        numpy.array([0.1, 0.2]), numpy.array([0.0, 90.0, 180.0, 270.0])
    )
    supplied_spectral_values = numpy.arange(8, dtype=float).reshape(2, 4)

    fill = fake_restart_file._fill_spectrum(
        "user_defined", spectral_values=supplied_spectral_values
    )

    assert numpy.array_equal(fill, supplied_spectral_values)


def test_fill_spectrum_user_defined_wrong_shape_raises():
    fake_restart_file = _FakeRestartFile(
        numpy.array([0.1, 0.2]), numpy.array([0.0, 90.0, 180.0, 270.0])
    )

    try:
        fake_restart_file._fill_spectrum(
            "user_defined", spectral_values=numpy.zeros((3, 3))
        )
        assert False, "expected a ValueError for a mismatched spectral_values shape"
    except ValueError:
        pass


def test_fill_spectrum_unknown_fill_type_raises():
    fake_restart_file = _FakeRestartFile(
        numpy.array([0.1, 0.2]), numpy.array([0.0, 90.0, 180.0, 270.0])
    )

    try:
        fake_restart_file._fill_spectrum("not_a_real_fill_type")
        assert False, "expected a ValueError for an unknown fill_type"
    except ValueError:
        pass


def test_fill_spectrum_gaussian_matches_target_significant_wave_height():
    frequency = numpy.geomspace(0.035, 0.5, 30)
    direction = numpy.linspace(0, 360, 36, endpoint=False)
    fake_restart_file = _FakeRestartFile(frequency, direction)

    target_significant_wave_height = 2.0
    fill = fake_restart_file._fill_spectrum(
        "gaussian",
        peak_frequency=0.1,
        frequency_spread=0.01,
        mean_direction=90.0,
        directional_spreading_power=4,
        significant_wave_height=target_significant_wave_height,
    )

    frequency_bin_width = midpoint_rule_step(frequency)
    direction_bin_width = 360.0 / len(direction)
    total_variance = (
        numpy.sum(fill * frequency_bin_width[:, None]) * direction_bin_width
    )
    resulting_significant_wave_height = 4.0 * numpy.sqrt(total_variance)

    assert (
        numpy.abs(resulting_significant_wave_height - target_significant_wave_height)
        < 1e-3
    )
    assert numpy.argmax(numpy.sum(fill, axis=1)) == numpy.argmin(
        numpy.abs(frequency - 0.1)
    )


def test_fill_spectrum_jonswap_peaks_near_peak_frequency():
    frequency = numpy.geomspace(0.035, 0.5, 60)
    direction = numpy.linspace(0, 360, 36, endpoint=False)
    fake_restart_file = _FakeRestartFile(frequency, direction)

    fill = fake_restart_file._fill_spectrum(
        "jonswap",
        peak_frequency=0.1,
        alpha=0.01,
        gamma=3.3,
        sigma_a=0.07,
        sigma_b=0.09,
        mean_direction=180.0,
        directional_spreading_power=2,
    )

    assert numpy.all(fill >= 0.0)
    assert numpy.argmax(numpy.sum(fill, axis=1)) == numpy.argmin(
        numpy.abs(frequency - 0.1)
    )


def test_fill_missing_spectra_leaves_valid_points_untouched_and_fills_missing():
    frequency = numpy.array([0.1, 0.2])
    direction = numpy.array([0.0, 90.0, 180.0, 270.0])

    valid_values = numpy.ones((2, 4)) * 5.0
    values = numpy.stack([valid_values, numpy.full((2, 4), numpy.nan)], axis=0)
    spectrum = _make_spectrum(frequency, direction, values)
    fake_restart_file = _FakeRestartFile(frequency, direction)

    filled_spectrum = fake_restart_file.fill_missing_spectra(spectrum, fill_type="calm")
    filled_values = filled_spectrum.directional_variance_density.values

    assert numpy.array_equal(filled_values[0], valid_values)
    assert numpy.all(filled_values[1] == 0.0)
    # the input spectrum passed in is not mutated
    assert numpy.all(numpy.isnan(spectrum.directional_variance_density.values[1]))


def test_fill_missing_spectra_is_a_no_op_when_nothing_is_missing():
    frequency = numpy.array([0.1, 0.2])
    direction = numpy.array([0.0, 90.0, 180.0, 270.0])

    valid_values = numpy.ones((1, 2, 4)) * 5.0
    spectrum = _make_spectrum(frequency, direction, valid_values)
    fake_restart_file = _FakeRestartFile(frequency, direction)

    filled_spectrum = fake_restart_file.fill_missing_spectra(spectrum, fill_type="calm")

    assert numpy.array_equal(
        filled_spectrum.directional_variance_density.values, valid_values
    )
