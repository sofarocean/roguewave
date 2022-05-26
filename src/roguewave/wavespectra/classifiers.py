from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D
from .wavespectrum import WaveSpectrum
from .estimators import spec1d_from_spec2d
from .parametric import pierson_moskowitz_frequency
from .partitioning import merge_partitions, find_label_closest_partition
import typing
import numpy
from . import logger
import logging


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


def find_matching_partition(source_spectrum: WaveSpectrum2D,
                            spectra: typing.Dict[int, WaveSpectrum2D]) -> \
typing.Tuple[int, float]:
    correlation = numpy.zeros((len(spectra),))
    labels = numpy.zeros((len(spectra),), dtype='int64')

    # source = source_spectrum.variance_density[:] - \
    #          numpy.mean(source_spectrum.variance_density[:])
    source = source_spectrum.variance_density.flatten()

    ii = -1
    for label, target_spectrum in spectra.items():
        ii += 1
        labels[ii] = label
        if target_spectrum is not None:
            # target = target_spectrum.variance_density[:] - numpy.mean(
            #     target_spectrum.variance_density[:])
            target = target_spectrum.variance_density.flatten()
            correlation[ii] = \
                numpy.nansum( source*target)
                #numpy.corrcoef(source.flatten(), target.flatten(), )[0, 1]
        else:
            correlation[ii] = 0

    i = numpy.nanargmax(correlation)
    return labels[i], correlation[i]


def find_all_matching_partitions(
        source_spectra: typing.Dict[int, WaveSpectrum2D],
        target_spectra: typing.Dict[int, WaveSpectrum2D], match_threshold=0.7):
    correlation = {}

    for label, source_spectrum in source_spectra.items():
        correlation[label] = find_matching_partition(source_spectrum,
                                                     target_spectra)
    return correlation


def check_for_orphaned_label(paths,label,proximity,curr):
    if paths[label][0] is None and paths[label][1] is None:
        target = find_label_closest_partition(label,
                                          proximity, curr)
        merge_partitions(label, target, curr, proximity)
        paths.pop(label)

def link_partitions(prev: typing.Dict[int, WaveSpectrum2D],
                    curr: typing.Dict[int, WaveSpectrum2D],
                    next: typing.Dict[int, WaveSpectrum2D],
                    proximity: typing.Dict[int, typing.List[int]],
                    is_first=False,
                    is_last=False,
                    threshold=0.1):
    curr_to_prev = find_all_matching_partitions(curr, prev)


    labels = list(curr.keys())

    start = {}
    end = {}
    paths = {}


    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\t correlation between current, previous and next")
        logger.debug(f"\t\t strength prev curr next strength")
        for label in labels:
            prev_label, strength_prev = curr_to_prev[label]
            logger.debug(
                f"\t\t{strength_prev:4.2e} {prev_label:04d} {label:04d} "
            )

    for label in labels:
        prev_label, strength_prev = curr_to_prev[label]

        prev_in_start = prev_label in start

        # not a path with same start point
        if prev_in_start:
            if strength_prev > start[prev_label]['correlation']:
                # replace
                paths[ start[prev_label]['label'] ][0] = None
                check_for_orphaned_label( paths, start[prev_label]['label'],proximity,curr  )
                start[prev_label] = {'label': label,
                                     'correlation': strength_prev}
            else:
                prev_label = None
        else:
            start[prev_label] = {'label': label,'correlation': strength_prev}


        paths[label] = [prev_label, None]

    # Step two- check for the threshold
    labels = list(paths.keys())
    for label in labels:
        path = paths[label]
        if path[0] is not None:
            if start[path[0]]['correlation'] < threshold:
                path[0] = None

        #check_for_orphaned_label(paths,label,proximity,curr)

    return paths


def link_partitions_old(prev: typing.Dict[int, WaveSpectrum2D],
                    curr: typing.Dict[int, WaveSpectrum2D],
                    next: typing.Dict[int, WaveSpectrum2D],
                    proximity: typing.Dict[int, typing.List[int]],
                    is_first=False,
                    is_last=False,
                    threshold=0.1):
    curr_to_prev = find_all_matching_partitions(curr, prev)
    curr_to_next = find_all_matching_partitions(curr, next)

    labels = list(curr.keys())

    start = {}
    end = {}
    paths = {}


    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\t correlation between current, previous and next")
        logger.debug(f"\t\t strength prev curr next strength")
        for label in labels:
            prev_label, strength_prev = curr_to_prev[label]
            next_label, strength_next = curr_to_next[label]
            logger.debug(
                f"\t\t{strength_prev:4.2e} {prev_label:04d} {label:04d} {next_label:04d} {strength_next:4.2e} "
            )

    for label in labels:
        prev_label, strength_prev = curr_to_prev[label]
        next_label, strength_next = curr_to_next[label]

        prev_in_start = prev_label in start
        next_in_end = next_label in end

        if prev_in_start and next_in_end:
            # already assigned - likely a path with the same start and end
            # points already exists
            if start[prev_label]['label'] == end[next_label]['label']:

                # check if the current candiate has stronger correlations
                if strength_prev * strength_next > start[prev_label][
                    'correlation'] * end[next_label]['correlation']:
                    # if so replace
                    label_to_remove = start[prev_label]['label']
                    start[prev_label] = {'label': label,
                                         'correlation': strength_prev}
                    end[next_label] = {'label': label,
                                       'correlation': strength_next}
                    paths.pop(label_to_remove)
                    paths[label] = [prev_label,next_label]
                else:
                    # if not- remove the current candiate path
                    label_to_remove = label

                target = find_label_closest_partition(label_to_remove,
                                                      proximity, curr)
                merge_partitions(label_to_remove, target, curr, proximity)
                continue
            else:
                pass

        # not a path with same start and end point
        if prev_in_start:
            if strength_prev > start[prev_label]['correlation']:
                # replace
                paths[ start[prev_label]['label'] ][0] = None
                check_for_orphaned_label( paths, start[prev_label]['label'],proximity,curr  )
                start[prev_label] = {'label': label,
                                     'correlation': strength_prev}
            else:
                prev_label = None
        else:
            start[prev_label] = {'label': label,'correlation': strength_prev}

        if next_in_end:
            if strength_next > end[next_label]['correlation']:
                # replace
                paths[ end[next_label]['label'] ][1] = None
                check_for_orphaned_label( paths, end[next_label]['label'],proximity,curr  )
                end[next_label] = {'label': label,
                                   'correlation': strength_prev}
            else:
                next_label = None
        else:
            end[next_label] = {'label': label, 'correlation': strength_next}

        if (next_label is None) and (prev_label is None):
            target = find_label_closest_partition(label,
                                                  proximity, curr)
            merge_partitions(label, target, curr, proximity)
        else:
            paths[label] = [prev_label, next_label]

    # Step two- check for the threshold
    labels = list(paths.keys())
    for label in labels:
        path = paths[label]
        if path[0] is not None:
            if start[path[0]]['correlation'] < threshold:
                path[0] = None

        if is_last and path[0] is None:
            path[1] = None

        if path[1] is not None:
            if end[path[1]]['correlation'] < threshold:
                path[1] = None

        if is_first and path[1] is None:
            path[0] = None

        check_for_orphaned_label(paths,label,proximity,curr)

    return paths