import numpy
from datetime import timedelta
from typing import List, Dict
from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput
from .wavespectrum import WaveSpectrum, extract_bulk_parameter
from .classifiers import link_partitions
from pandas import DataFrame

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


def link_and_merge(spectra: List[Dict[int, WaveSpectrum2D]],
                   proximity: List[Dict[int,List[int]]], threshold=0.7)->Dict[int,List[WaveSpectrum2D]]:
    """


    :param spectra:
    :param proximity:
    :param threshold:
    :return:
    """

    # List of all wave fields we have identified
    fields = {} # type: Dict[int,List[WaveSpectrum2D]]

    # list of wave fields that are currently active. The _key_ refers to the
    # partition label in the preceeding spectrum, the entry is the label linking
    # to the current active fields.
    active = {} # type: Dict[int, int]

    # Initialize the field counter
    field_label_counter = -1
    for index, curr in enumerate(spectra):

        # Identify indices of the previous and next spectrum in the list, and
        # ensure that we do not go out of bounds.
        iu = min( index + 1 , len(spectra)-1 )
        id = max( index - 1 , 0)

        # find the local path connecting partitions:
        local_paths = link_partitions( spectra[id], curr, spectra[iu],
                                       proximity[index],threshold=threshold)

        for prev_label,curr_label,next_label in local_paths:
            # Get a path, that consistof:
            #    prev_label: the label associated with the partition in the
            #                previous (in time) spectrum.
            #    curr_label: the label associated with the partition in the
            #                current (in time) spectrum.
            #    next_label: the label associated with the partition in the
            #                next (in time) spectrum.
            if prev_label in active:
                # if the previous label is part of the active set - add the
                # current partition to the that wave field
                field_label = active[prev_label]
                fields[field_label].append( curr[curr_label] )

                # pop the previous label from the active set...
                active.pop(prev_label)
                if next_label is not None:
                    # and if the current path is connected to the next path
                    # replace it with the current path
                    active[curr_label] = field_label
                else:
                    # This is the end of the road for this partition.
                    pass
            else:
                # if not in the active set, create a new label
                field_label_counter += 1
                fields[field_label_counter] = [curr[curr_label]]

                if next_label is not None:
                    active[curr_label] = field_label_counter
                else:
                    # This spectrun is orphaned,
                    print(prev_label,curr_label,next_label)
                    print('should not happen')

    return fields

def bulk_parameters_partitions( partitions:Dict[int,List[WaveSpectrum2D]] ):
    bulk = {}
    for label, partition in partitions.items():
        df = DataFrame()
        for variable in WaveSpectrum.bulk_properties:
            df[variable] = extract_bulk_parameter(variable, partition)
        bulk[label] = df
    return bulk