from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D
from .wavespectrum import WaveSpectrum
from .estimators import spec1d_from_spec2d
from .parametric import pierson_moskowitz_frequency
from .partitioning import merge_partitions, find_label_closest_partition
import typing
import numpy


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
                            spectra: typing.Dict[int, WaveSpectrum2D])->typing.Tuple[int,float]:
    correlation =numpy.zeros( (len(spectra), ))
    labels = numpy.zeros( (len(spectra), ),dtype='int64')

    source = source_spectrum.variance_density[:] - \
             numpy.mean(source_spectrum.variance_density[:])
    source = source.flatten()

    ii = -1
    for label, target_spectrum in spectra.items():
        ii+=1
        labels[ii] = label
        if target_spectrum is not None:
            target = target_spectrum.variance_density[:] - numpy.mean(
                target_spectrum.variance_density[:])
            target = target.flatten()
            correlation[ii] = \
                numpy.corrcoef(source.flatten(), target.flatten(), )[0, 1]
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


def link_partitions(prev: typing.Dict[int, WaveSpectrum2D],
                    curr: typing.Dict[int, WaveSpectrum2D],
                    next: typing.Dict[int, WaveSpectrum2D],
                    proximity: typing.Dict[int,typing.List[int]],
                    threshold=0.1):

    curr_to_prev = find_all_matching_partitions( curr, prev)
    curr_to_next = find_all_matching_partitions( curr, next)

    labels = list(curr.keys())

    connections = {label_prev: { label_next: [   ] for label_next in next  }
                   for label_prev in prev
                   }

    def filter_for_dual_paths(connections, curr, proximity):
        for prev_label, next_dict in connections.items():
            for next_label, label_list in next_dict.items():
                if len(label_list)<2:
                    continue

                labels = [ x[0] for x in label_list ]
                icorr = numpy.argmax([ x[1]*x[2] for x in label_list ])

                for label in labels:
                    if labels[icorr] == label:
                        continue

                    target = find_label_closest_partition(label,proximity,curr)
                    merge_partitions(label,target,curr,proximity)
                connections[prev_label][next_label] =

    for label in labels:
        prev_label, strength_prev = curr_to_prev[label]
        next_label, strength_next = curr_to_next[label]

        connections[ prev_label ][next_label].append( (label , strength_prev, strength_next ))



    paths = []
    end_point = { key: [] for key in next.keys() }
    start_point = {key: [] for key in prev.keys()}
    jj = -1
    for prev_label, next_dict in connections.items():
        for next_label, label_list in next_dict.items():
            if not label_list:
                continue
            jj +=1

            if len(label_list) == 1:
                #
                label, strength_prev, strength_next = label_list[0]
                if strength_prev > threshold and strength_next > threshold:
                    paths.append( [prev_label, label, next_label] )
                    end_point[next_label].append( ( strength_next, len(paths)-1 ) )
                    start_point[prev_label].append(
                        (strength_prev, len(paths) - 1))
                elif strength_prev < threshold and strength_next > threshold:
                    paths.append( [None, label, next_label] )
                    end_point[next_label].append((strength_next, len(paths) - 1))
                    
                elif strength_prev > threshold and strength_next < threshold:
                    paths.append( [prev_label, label, None] )
                    start_point[prev_label].append(
                        (strength_prev, len(paths) - 1))
                else:
                    # Orphaned - lets delete
                    target = find_label_closest_partition(label,proximity,curr)
                    merge_partitions(label, target, curr, proximity)
            else:
                # dual paths
                labels = [ x[0] for x in label_list ]
                icorr = numpy.argmax([ x[1]*x[2] for x in label_list ])
                paths.append([prev_label, labels[icorr], next_label])

                end_point[next_label].append((strength_next, len(paths) - 1))
                start_point[prev_label].append(
                    (strength_prev, len(paths) - 1))

                for label in labels:
                    if labels[icorr] == label:
                        continue

                    target = find_label_closest_partition(label,proximity,curr)
                    merge_partitions(label,target,curr,proximity)


    to_pop = []
    for label, paths_ending in end_point.items():
        if len(paths_ending) > 1:
            # multiple paths are ending here, lets only keep the one with the
            # strongest correlation
            imax = numpy.argmax(numpy.array([ x[0] for x in paths_ending ]))

            for ii, entry in enumerate(paths_ending):
                # for all the paths that end here do
                if ii == imax:
                    continue

                path_index = entry[1]
                path = paths[path_index]
                path[2] = None

                if not path[0]:
                    # Orphaned - lets delete
                    source = path[1]
                    target = find_label_closest_partition(source, proximity,
                                                          curr)
                    merge_partitions(source, target, curr, proximity)
                    to_pop.append( path_index )

    for index in sorted(to_pop,reverse=True):
        paths.pop(index)

    to_pop = []
    for label, paths_starting in start_point.items():
        if len(paths_starting) > 1:
            # multiple paths are starting here, lets only keep the one with the
            # strongest correlation
            imax = numpy.argmax(numpy.array([x[0] for x in paths_starting]))

            for ii, entry in enumerate(paths_starting):
                # for all the paths that end here do
                if ii == imax:
                    continue

                path_index = entry[1]
                path = paths[path_index]
                path[0] = None

                if not path[2]:
                    # Orphaned - lets delete
                    source = path[1]
                    target = find_label_closest_partition(source, proximity,
                                                          curr)
                    merge_partitions(source, target, curr, proximity)
                    to_pop.append(path_index)

    for index in sorted(to_pop, reverse=True):
        paths.pop(index)

    return paths


def prune_multiple_start_or_exit( is_start:bool ,paths, list_of_converging_paths: typing.Dict[int,typing.Tuple[int,int]] ):
    to_pop = []
    for label, paths_converging in list_of_converging_paths.items():
        if len(paths_converging) > 1:
            # multiple paths are ending here, lets only keep the one with the
            # strongest correlation
            imax = numpy.argmax(numpy.array([x[0] for x in paths_converging]))

            for ii, entry in enumerate(paths_converging):
                # for all the paths that end here do
                if ii == imax:
                    continue

                path_index = entry[1]
                path = paths[path_index]
                path[2] = None

                if not path[0]:
                    # Orphaned - lets delete
                    source = path[1]
                    target = find_label_closest_partition(source, proximity,
                                                          curr)
                    merge_partitions(source, target, curr, proximity)
                    to_pop.append(path_index)

    for index in sorted(to_pop, reverse=True):
        paths.pop(index)

