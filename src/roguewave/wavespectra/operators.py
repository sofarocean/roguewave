import numpy
from typing import List
from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from datetime import timedelta

def spectrum1D_time_filter(spectra: List[WaveSpectrum1D],
                         window: numpy.ndarray = None,
                         maximum_time_delta: timedelta = None) -> List[
    WaveSpectrum1D]:
    number_of_spectra = len(spectra)

    if not window:
        window = numpy.array([0.25, 0.5, 0.25])

    if not maximum_time_delta:
        maximum_time_delta = timedelta(hours=2)

    window_length = len(window)
    window_midpoint = window_length // 2

    # shallow copy of the list
    spectra = spectra.copy()
    first = spectra[0]
    last = spectra[-1]

    for ii in range(0, window_midpoint):
        spectra.insert(0, first)
        spectra[0].timestamp = spectra[1].timestamp - maximum_time_delta

    for ii in range(window_midpoint + 1, window_length):
        spectra.append(last)
        spectra[-1].timestamp = spectra[-1].timestamp + maximum_time_delta

    output = []
    for index in range(0, number_of_spectra):
        istart = index
        iend = index + window_length

        timedeltas = numpy.array([
            next.timestamp - cur.timestamp for cur, next in
            zip(spectra[istart: iend - 1], spectra[istart + 1: iend])
        ])

        raw_spectrum = spectra[istart + window_midpoint].copy()
        if numpy.any(timedeltas > maximum_time_delta):
            output.append(raw_spectrum)
            continue

        e = sum(w * spectrum.e for w, spectrum in
                zip(window, spectra[istart: iend]))
        a1 = sum(w * spectrum.a1 * spectra[index].e for w, spectrum in
                 zip(window, spectra[istart: iend])) / e
        b1 = sum(w * spectrum.b1 * spectra[index].e for w, spectrum in
                 zip(window, spectra[istart: iend])) / e
        a2 = sum(w * spectrum.a2 * spectra[index].e for w, spectrum in
                 zip(window, spectra[istart: iend])) / e
        b2 = sum(w * spectrum.b2 * spectra[index].e for w, spectrum in
                 zip(window, spectra[istart: iend])) / e

        output.append(WaveSpectrum1D(WaveSpectrum1DInput(
            frequency=raw_spectrum.frequency,
            varianceDensity=e,
            timestamp=raw_spectrum.timestamp,
            latitude=raw_spectrum.latitude,
            longitude=raw_spectrum.longitude,
            a1=a1, b1=b1,
            a2=a2, b2=b2))
        )
    return spectra


def spectrum2D_time_filter(spectra: List[WaveSpectrum2D],
                         window: numpy.ndarray = None,
                         maximum_time_delta: timedelta = None) -> List[
    WaveSpectrum2D]:
    number_of_spectra = len(spectra)

    if not window:
        window = numpy.array([0.25, 0.5, 0.25])

    if not maximum_time_delta:
        maximum_time_delta = timedelta(hours=2)

    window_length = len(window)
    window_midpoint = window_length // 2

    # shallow copy of the list
    spectra = spectra.copy()
    first = spectra[0]
    last = spectra[-1]

    for ii in range(0, window_midpoint):
        spectra.insert(0, first)
        spectra[0].timestamp = spectra[1].timestamp - maximum_time_delta

    for ii in range(window_midpoint + 1, window_length):
        spectra.append(last)
        spectra[-1].timestamp = spectra[-1].timestamp + maximum_time_delta

    output = []
    for index in range(0, number_of_spectra):
        istart = index
        iend = index + window_length

        timedeltas = numpy.array([
            next.timestamp - cur.timestamp for cur, next in
            zip(spectra[istart: iend - 1], spectra[istart + 1: iend])
        ])

        raw_spectrum = spectra[istart + window_midpoint].copy()
        if numpy.any(timedeltas > maximum_time_delta):
            output.append(raw_spectrum)
            continue

        variance_density = sum(w * spectrum.variance_density for w, spectrum in
                zip(window, spectra[istart: iend]))

        output.append(
            WaveSpectrum2D(
                WaveSpectrum2DInput(
                    frequency=raw_spectrum.frequency,
                    varianceDensity=variance_density,
                    timestamp=raw_spectrum.timestamp,
                    latitude=raw_spectrum.latitude,
                    longitude=raw_spectrum.longitude,
                    directions=raw_spectrum.direction
                )
            )
        )
    return spectra