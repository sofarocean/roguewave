from .spectrum2D import WaveSpectrum2D
from .wavespectrum import WaveSpectrum
from .estimators import spec1d_from_spec2d
from .parametric import pierson_moskowitz_frequency


def is_sea_spectrum(spectrum: WaveSpectrum) -> bool:
    """
    Identify whether or not it is a sea partion. Use 1D method for 2D
    spectra in section 3 of:

    Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
    Spectral partitioning and identification of wind sea and swell.
    Journal of atmospheric and oceanic technology, 26(1), 107-122.

    :return: boolean indicting it is sea
    """

    if isinstance(spectrum, WaveSpectrum2D):
        spectrum = spec1d_from_spec2d(spectrum)

    peak_index = spectrum.peak_index()
    peak_frequency = spectrum.frequency[peak_index]

    return pierson_moskowitz_frequency(
        peak_frequency, peak_frequency) <= spectrum.variance_density[
               peak_index]


def is_swell_spectrum(spectrum: WaveSpectrum) -> bool:
    """
    Identify whether or not it is a sea partion. Use 1D method for 2D
    spectra in section 3 of:

    Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
    Spectral partitioning and identification of wind sea and swell.
    Journal of atmospheric and oceanic technology, 26(1), 107-122.

    :return: boolean indicting it is sea
    """

    return not is_sea_spectrum(spectrum)


