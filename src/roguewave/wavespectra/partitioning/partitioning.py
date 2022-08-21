"""
Contents: Partitioning

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Partitioning routines that can be used to partition the spectrum.

Classes:

- `SeaSwellData`, return type that contains the partitioned data.
- `Partition`, data class that describes a partition

Functions:

- `neighbours`,

How To Use This Module
======================
(See the individual functions for details.)

1. Import it: ``import partitioning`` or ``from partitioning import ...``.
2.
"""

from roguewave.wavespectra import FrequencyDirectionSpectrum
import numpy
import numba.typed
from typing import Dict, List, Tuple
import numba
from scipy.ndimage import maximum_filter, generate_binary_structure

default_partition_config = {
    'minimumEnergyFraction': 0.001,
    'minimumDensityRatio': 0.
}

NOT_ASSIGNED = -1


@numba.njit(cache=True)
def floodfill(spectral_density: numpy.ndarray, min_val=0.0) -> Tuple[
    numpy.ndarray, Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
    """
    Flood fill algorithm. We try to find the region that belongs to a peak 
    according to inverse watershed.The implementation has been optimized for 
    use with numba.

    :param spectral_density: 2d-ndarray, first dimension frequency, second direction
    :param min_val: minimum value we consider in the algorithm.

    :return:

    We start at a local peak indicated by [peak_frequency_index]
    and [peak_direction_index], label that peak with the given [label]
    """

    number_of_directions = spectral_density.shape[1]
    number_of_frequencies = spectral_density.shape[0]

    proximate_partitions = {}  # type: Dict[int,List[int]]

    partition_label = numpy.zeros(
        (number_of_frequencies, number_of_directions),
        dtype='int32') + NOT_ASSIGNED

    current_label = 0
    assigned = False
    for start_frequency_index in range(0, number_of_frequencies):
        for start_direction_index in range(0, number_of_directions):
            if spectral_density[
                start_frequency_index, start_direction_index] == min_val:
                assigned = True
                partition_label[
                    start_frequency_index, start_direction_index] = current_label

    if assigned:
        proximate_partitions[current_label] = numpy.array([0], dtype='int64')

    #
    # We loop over all cells in the spectrum...
    peak_indices = {}
    # delta = numpy.zeros( 8 )
    ii = numpy.zeros(1000, dtype='int64')
    jj = numpy.zeros(1000, dtype='int64')

    neighbour_direction_offset = numpy.array(
        (0, 0, - 1, - 1, - 1, + 1, + 1, + 1), dtype='int64')
    neighbour_frequency_offset = numpy.array(
        (- 1, + 1, - 1, 0, + 1, - 1, 0, + 1), dtype='int64')
    neighbour_frequency_indices = numpy.zeros(8, dtype='int64')
    neighbour_direction_indices = numpy.zeros(8, dtype='int64')

    for start_frequency_index in range(0, number_of_frequencies):
        for start_direction_index in range(0, number_of_directions):

            # if the cell is already assigned to a peak during a previous ascent
            # from another staring point we can skip the current cell.
            if partition_label[
                start_frequency_index, start_direction_index] > NOT_ASSIGNED:
                continue

            # Initialize the path. We keep track of the visited cells in an
            # ascent by denoting their indices in a list.
            # ii = [start_frequency_index]
            # jj = [start_direction_index]
            ii[0] = start_frequency_index
            jj[0] = start_direction_index
            number_of_points_in_path = 1

            # In addition- as we climb the hill we denote any partitions we
            # happen to neighbour
            proximate_partitions_work = []

            while True:
                # get the last cell in the path
                direction_index = jj[number_of_points_in_path - 1]  # jj[-1]
                frequency_index = ii[number_of_points_in_path - 1]  # ii[-1]

                if (partition_label[
                    frequency_index, direction_index] > NOT_ASSIGNED):
                    label = partition_label[frequency_index, direction_index]
                    proximate_partitions[label] = \
                        numpy.append(proximate_partitions[label],
                                     proximate_partitions_work)

                    for index in range(0, number_of_points_in_path):
                        partition_label[ii[index], jj[index]] = label
                    break

                neighbour_frequency_indices[
                :] = neighbour_frequency_offset + frequency_index
                neighbour_direction_indices[:] = (
                                                         neighbour_direction_offset + direction_index) % number_of_directions

                # Get the value of the energy density at the current cell
                node_value = spectral_density[frequency_index, direction_index]

                # Here we calculate the differences in energy density from the
                # current cell to each of it neighbours.
                # delta[:] = -numpy.inf
                all_smaller_than_zero = True
                index_max_delta = -1
                max_delta = -numpy.inf
                for index, ifreq, idir in zip(
                        range(0, len(neighbour_frequency_indices)),
                        neighbour_frequency_indices,
                        neighbour_direction_indices):

                    if ifreq < 0 or ifreq >= number_of_frequencies:
                        continue

                    # Calculate the delta for each neighbour
                    delta = (spectral_density[ifreq, idir] - node_value)
                    if delta > max_delta:
                        index_max_delta = index
                        max_delta = delta

                    if delta > 0:
                        all_smaller_than_zero = False

                    if partition_label[ifreq, idir] > 0:
                        proximate_partitions_work.append(
                            partition_label[ifreq, idir])

                if all_smaller_than_zero:
                    # this is a peak
                    current_label += 1
                    label = current_label
                    proximate_partitions[label] = numpy.array(
                        proximate_partitions_work, dtype='int64')
                    peak_indices[label] = (frequency_index, direction_index)

                    for index in range(0, number_of_points_in_path):
                        partition_label[ii[index], jj[index]] = label

                    break
                else:
                    # steepest_index = numpy.argmax(delta)
                    ii[number_of_points_in_path] = neighbour_frequency_indices[
                        index_max_delta]
                    jj[number_of_points_in_path] = neighbour_direction_indices[
                        index_max_delta]
                    number_of_points_in_path += 1

                    if number_of_points_in_path == len(ii):
                        # because we preallocate the path arrays we have to
                        # make sure they fit- and otherwise extend if need be
                        _ii = numpy.zeros(2 * len(ii), dtype='int64')
                        _jj = numpy.zeros(2 * len(ii), dtype='int64')
                        _ii[0:number_of_points_in_path] = ii
                        _jj[0:number_of_points_in_path] = jj
                        ii = _ii
                        jj = _jj

                    # ii.append(neighbour_frequency_indices[steepest_index])
                    # jj.append(neighbour_direction_indices[steepest_index])
                    continue
        #

    for label in proximate_partitions:
        proximate_partitions[label] = numpy.unique(proximate_partitions[label])
        mask = proximate_partitions[label] == label
        proximate_partitions[label] = proximate_partitions[label][~mask]

    for label_source, item in proximate_partitions.items():
        for label_target in item:
            if label_source == label_target:
                continue

            if label_source not in proximate_partitions[label_target]:
                proximate_partitions[label_target] = numpy.append(
                    proximate_partitions[label_target], label_source)

    return partition_label, proximate_partitions, peak_indices


@numba.njit(cache=True)
def find_label_closest_partition(label,
                                 proximate_labels: List[int],
                                 number_of_directions: int,
                                 peak_indices: Dict[int, Tuple[int, int]]):
    """
    Find the partition whose peak is "closest" to the partition under consideration
    Only proximate partitions are considered. Basically: find the partition whose
    maximum is closest to the peak of the current partition.

    :param label:
    :param proximate_labels:
    :param number_of_directions:
    :param peak_indices:

    :return:
    """

    if not proximate_labels:
        return None

    # Initialize the numpy arrays for distance and indices.
    distance = numpy.zeros((len(proximate_labels),))
    indices = numpy.zeros((len(proximate_labels),), dtype='int64')

    ifreq, idir = peak_indices[label]
    for ii, local_label in enumerate(proximate_labels):
        delta_freq = peak_indices[local_label][0] - ifreq
        delta_dir = (peak_indices[local_label][
                         1] - idir - number_of_directions // 2) % number_of_directions + number_of_directions // 2
        distance[ii] = numpy.sqrt(delta_freq ** 2 + delta_dir ** 2)
        indices[ii] = local_label

    return indices[numpy.argmin(distance)]


def filter_for_low_energy(partitions: Dict[int, FrequencyDirectionSpectrum],
                          proximity: Dict[int, List[int]],
                          peak_indices: Dict[int, Tuple[int, int]],
                          total_energy,
                          config):
    """
    Remove partitions that contain only a small fraction of the total energy
    in the spectrum.

    :param partitions: Dictionary of spectra, keys are the labels identifying
                       the partition under consideration

    :param proximity: Dictionary that denotes for each partition identified by
                      its label (dict key) a list of partitions that are direct
                      neighbours

    :param peak_indices: Dictionary that for each partition identified by its
                         label (dict key) the frequency and direction index of
                         the peak value

    :param total_energy: Total energy (variance) in the wave spectrum

    :param config: Configuration dictionary

    :return: None (function has side effects)
    """

    # If there is only one partition, there is no need to filter, return
    if len(partitions) == 1:
        return

    # loop over all the partition labels
    for label in list(partitions.keys()):

        # This looks weird- but is needed. The dictionary is changed and a label
        # may no longer exist.
        if label not in partitions:
            continue

        # get the partition associated with the label
        partition = partitions[label]

        # Check if the partition contains sufficient energy
        if (partition.m0() < config['minimumEnergyFraction'] * total_energy):
            #
            # If not, Merge with the closest partition. First find what is the
            # closest neighbour.
            if partition.hm0() == 0:
                closest_label = proximity[label][0]
            else:
                # Note we have to pass a numba typed list to avoid the deprecation
                # warning on general python lists.

                if not len(proximity[label]) == 0:
                    closest_label = find_label_closest_partition(
                        label, numba.typed.List(proximity[label]),
                        len(partition.direction), peak_indices)
                else:
                    closest_label = None

            # There is an edge case where there are no neighbours to merge with
            # hence the check
            if closest_label is not None:
                # if a closest partition is found- lets merge with that
                # partition.
                merge_partitions(label, closest_label, partitions,
                                 proximity)


def merge_partitions(
        source_label,
        target_label,
        partitions: Dict[int, FrequencyDirectionSpectrum],
        proximity: Dict[int, List[int]]) -> None:
    """
    Merge one partition into another partition. (**Has side-effects**)
    :param source_index: source index of the partition to merge in the list
    :param target_index: target index of the partition that is merged into
    :param partitions: the list of partitions
    :param partition_label_array: 2D array of labels indicating to what partition
    the cel belongs

    :return: None
    """

    # Add the source to the target spectrum.
    partitions[target_label] = partitions[target_label] + partitions[
        source_label]

    # Remove the source from the partition dict
    partitions.pop(source_label)

    # The merge is now complete, but we need to update the adjacency list that
    # denotes which partitions border one-another since anything that bordered
    # the source partition will now border the target partition. Furter, we need
    # to remove references to the source partition in the adjacency list.

    # loop over all partitions bordering the source partition
    for proximate_labels_to_source_label in proximity[source_label]:

        # for the neighbour, get its list of partitions that border it.
        proximate_to_update = proximity[proximate_labels_to_source_label]

        # find the index in this list that refers to the source
        index = proximate_to_update.index(source_label)

        # If this is the target partition, merely remove the reference to the
        # source partition from the list
        if proximate_labels_to_source_label == target_label:
            proximity[target_label].pop(index)
            continue

        #
        # Make sure the partition listed as bordering the target partition.
        if proximate_labels_to_source_label not in proximity[target_label]:
            proximity[target_label].append(proximate_labels_to_source_label)

        # If the target already is listed as bordering, merely remove the
        # reference of the source from the list
        if target_label in proximate_to_update:
            proximate_to_update.pop(index)
        else:
            # if it does not border, update the reference to the source with
            # a reference to the target.
            proximate_to_update[index] = target_label

    # We can now safely remove the list from the proximity dict.
    proximity.pop(source_label)
    return None


def partition_spectrum(spectrum: FrequencyDirectionSpectrum, config=None) -> \
        Tuple[Dict[int, FrequencyDirectionSpectrum], Dict[
            int, List[int]]]:
    """
    Create partitioned spectra from a given 2D wavespectrum
    :param spectrum: 2D wavespectrum
    :param config: configuration parameters - see "default_partition_config"
    default variable.

    :return:
    """

    # Update config if provided
    config = default_partition_config | (config if config else {})

    # Make sure there are no NaN (undifined) values
    density = spectrum.spectral_values.values
    density[numpy.isnan(density)] = 0.0

    # Get partition descriptions using floodfill. This returns an array that
    # labels for each cell in the spectrum to which partition it belongs, an
    # ordered list with the label as index containing which partitions are
    # adjacent to the current partition and the labels.
    partition_label, proximate, peak_indices = floodfill(density)

    # We need to convert back to a list, and we need to do it in a somewhat
    # roundabout way as numba does not support dicts of lists (note, proximate
    # currently is a "numba" dictionary of numpy arrays)
    proximate_partitions = {}
    for label in proximate:
        proximate_partitions[label] = list(proximate[label])

    # Create Partition objects given the label array from the floodfill.
    partitions = {}

    # Create a dict of all draft partitions
    for label, proximity_list in proximate_partitions.items():
        if label == 0:
            continue
        mask = partition_label == label
        partitions[label] = spectrum.extract(mask)
    #
    # # merge low energy partitions
    filter_for_low_energy(partitions, proximate_partitions, peak_indices,
                         spectrum.m0(), config)

    # Return the partitions and the label array.
    return partitions, proximate_partitions


def partition_marker_array(
        partitions: Dict[int, FrequencyDirectionSpectrum]) -> numpy.ndarray:
    a_label = list(partitions.keys())[0]
    marker_array = numpy.zeros(
        partitions[a_label].spectral_values.shape) + numpy.nan
    for label, partition in partitions.items():
        mask = ~partition.spectral_values.mask
        marker_array[mask] = label
    return marker_array


def sum_partitions(
        partitions: Dict[int,FrequencyDirectionSpectrum]) \
        -> FrequencyDirectionSpectrum:
    key = list(partitions.keys())[0]
    sum_spec = partitions[key].copy(deep=True)
    sum_spec['variance_density'] *= 0
    for _, spec in partitions.items():
        sum_spec = spec + sum_spec
    return sum_spec


def is_neighbour(partition_1:FrequencyDirectionSpectrum,
                 partition_2: FrequencyDirectionSpectrum) -> int:
    footprint = generate_binary_structure(partition_2.spectral_values.ndim, 2)

    mask_1 = maximum_filter(partition_1.spectral_values.filled(0),
                            footprint=footprint,
                            mode='wrap')
    mask_2 = partition_2.spectral_values.filled(0)

    return numpy.nansum(mask_1 * mask_2)
