from roguewave.wavespectra.spectrum2D import WaveSpectrum2D
from roguewave.wavespectra.wavespectrum import WaveSpectrum
from roguewave.wavespectra.estimators import spec1d_from_spec2d
from roguewave.wavespectra.parametric import pierson_moskowitz_frequency
from roguewave.wavespectra.estimators import convert_to_1d_spectrum
from typing import Union, overload, List


@overload
def is_sea_spectrum(spectrum:WaveSpectrum, coefficient = 1.0) -> bool: ...

@overload
def is_sea_spectrum(spectrum:List[WaveSpectrum], coefficient = 1.0) -> List[bool]: ...

@overload
def is_swell_spectrum(spectrum:WaveSpectrum, coefficient = 1.0) -> bool: ...

@overload
def is_swell_spectrum(spectrum:List[WaveSpectrum], coefficient = 1.0) -> List[bool]: ...


def is_sea_spectrum(spectrum: Union[WaveSpectrum, List[WaveSpectrum]],
                    coefficient=1.0) -> Union[bool, list[bool]]:
    """
    Identify whether or not it is a sea partion. Use 1D method for 2D
    spectra in section 3 of:

    Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
    Spectral partitioning and identification of wind sea and swell.
    Journal of atmospheric and oceanic technology, 26(1), 107-122.

    :return: boolean indicting it is sea
    """

    if not isinstance(spectrum,list):
        spectra = [spectrum]
        return_list = False
    else:
        spectra = spectrum
        return_list = True

    spectra = convert_to_1d_spectrum(spectra)

    output = []
    for _spectrum in spectra:
        peak_index = _spectrum.peak_index()
        peak_frequency = _spectrum.frequency[peak_index]

        output.append(coefficient * pierson_moskowitz_frequency(
            peak_frequency, peak_frequency) <= _spectrum.variance_density[
                   peak_index])

    if not return_list:
        return output[0]
    else:
        return output


def is_swell_spectrum(spectrum: Union[WaveSpectrum, List[WaveSpectrum]],
                      coefficient=1.0) -> Union[bool, list[bool]]:
    """
    Identify whether or not it is a sea partion. Use 1D method for 2D
    spectra in section 3 of:

    Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
    Spectral partitioning and identification of wind sea and swell.
    Journal of atmospheric and oceanic technology, 26(1), 107-122.

    :return: boolean indicting it is sea
    """

    return not is_sea_spectrum(spectrum, coefficient=coefficient)
