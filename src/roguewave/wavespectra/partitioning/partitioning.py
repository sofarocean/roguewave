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

from roguewave.wavespectra.spectrum2D import WaveSpectrum2D, \
    empty_spectrum2D_like
import numpy
import typing
import numba
from scipy.ndimage import maximum_filter, generate_binary_structure

default_partition_config = {
    'minimumEnergyFraction': 0.01,
    'minimumDensityRatio': 0.
}


@numba.njit(cache=True)
def neighbours(peak_direction_index: int, peak_frequency_index: int,
               number_of_directions: int, number_of_frequencies: int,
               diagonals=True) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Find all indices that correspond to the 2D neighbours of a given point
    with indices (i, j) in a rectangular raster spanned by frequencies and
    directions. There are potentially 8 neighbours:

            + (i+1,j-1)   + (i+1,   j)   +  (i+1,j+1)    ( + = neighbour, x is point )

            + (i  ,j-1)   x (i  ,   j)   +  (i  ,j+1)

            + (i-1,j-1)   + (i-1,   j)   +  (i-1,j+1)

    resulting in output:

            frequency_indices = [i  , i  , i-1, i-1, i-1, i+1, i+1, i+1]
            direction_indices = [j-1, j+1, j-1, j  , j+1, j-1, j  , j+1]

    Note that due to directional wrapping j-1 and j+1 always exist; but
    i+1 and i-1 may not. The ordering of the output is due to  ease of
    implementation

    :param peak_direction_index: central direction index
    :param peak_frequency_index: central frequency index
    :param number_of_directions: number of direction in the grid
    :param number_of_frequencies: number of frequencies in the grid
    :param diagonals: (optional), whether or not to include the diagonal indices
    as neighbours.
    :return: ( frequency_indices, direction_indices  )
    """

    # First we need to calculate the next/prev angles in frequency ann
    # direction space, noting that direction space is periodic.
    next_direction_index = (peak_direction_index + 1) % number_of_directions
    prev_direction_index = (peak_direction_index - 1) % number_of_directions
    prev_frequency_index = peak_frequency_index - 1
    next_frequency_index = peak_frequency_index + 1

    # Add the
    frequency_indices = [peak_frequency_index, peak_frequency_index]
    direction_indices = [prev_direction_index, next_direction_index]

    if diagonals:
        # The diagonals are included
        if peak_frequency_index > 0:
            # add the neighbours at the previous frequency
            frequency_indices += [prev_frequency_index, prev_frequency_index,
                                  prev_frequency_index]
            direction_indices += [prev_direction_index, peak_direction_index,
                                  next_direction_index]

        if peak_frequency_index < number_of_frequencies - 1:
            # add the neighbours at the next frequency
            frequency_indices += [next_frequency_index, next_frequency_index,
                                  next_frequency_index]
            direction_indices += [prev_direction_index, peak_direction_index,
                                  next_direction_index]
    else:
        # The diagonals are not included

        if peak_frequency_index > 0:
            # add the neighbours at the previous frequency
            frequency_indices += [prev_frequency_index]
            direction_indices += [peak_direction_index]
        if peak_frequency_index < number_of_frequencies - 1:
            # add the neighbours at the next frequency
            frequency_indices += [next_frequency_index]
            direction_indices += [peak_direction_index]

    return numpy.array(frequency_indices, dtype='int64'), numpy.array(
        direction_indices, dtype='int64')


NOT_ASSIGNED = -1


@numba.njit(cache=True)
def floodfill(frequency: numpy.ndarray, direction: numpy.ndarray,
              spectral_density: numpy.ndarray, min_val=0.0) -> typing.Tuple[
    numpy.ndarray, typing.Dict[int, typing.List[int]]]:
    """
    Flood fill algorithm. We try to find the region that belongs to a peak according
    to inverse watershed.

    :param spectral_density: 2d-ndarray, first dimension frequency, second direction
    :param partition_label: 2d-integer-ndarray, of same shape as [spectral_density].
    for each entry contains label to which the current entry belongs. Negative
    label is unasigned.

    :param peak_frequency_index: frequency index of local peak
    :param peak_direction_index: direction index of local peak
    :return:

    We start at a local peak indicated by [peak_frequency_index]
    and [peak_direction_index], label that peak with the given [label]


    """

    number_of_directions = spectral_density.shape[1]
    number_of_frequencies = spectral_density.shape[0]

    proximate_partitions = {}  # type: typing.Dict[int,typing.List[int]]
    # proximate_partitions = []

    partition_label = numpy.zeros(
        (number_of_frequencies, number_of_directions),
        dtype='int32') + NOT_ASSIGNED

    current_label = 0
    assigned = False
    for start_frequency_index in range(0, number_of_frequencies):
        for start_direction_index in range(0, number_of_directions):
            if spectral_density[start_frequency_index,start_direction_index]==min_val:
                assigned = True
                partition_label[start_frequency_index,start_direction_index] = current_label

    if assigned:
        proximate_partitions[current_label] = numpy.array([0], dtype='int64')

    for start_frequency_index in range(0, number_of_frequencies):
        for start_direction_index in range(0, number_of_directions):
            # if already assigned - continue
            if partition_label[
                start_frequency_index, start_direction_index] > NOT_ASSIGNED:
                continue

            ii = [start_frequency_index]
            jj = [start_direction_index]
            proximate_partitions_work = []
            while True:

                direction_index = jj[-1]
                frequency_index = ii[-1]

                neighbour_frequency_indices, neighbour_direction_indices = neighbours(
                    direction_index, frequency_index,
                    number_of_directions, number_of_frequencies)

                node_value = spectral_density[frequency_index, direction_index]

                delta = numpy.zeros_like(neighbour_frequency_indices,dtype='float64')
                for index, ifreq, idir in zip(
                        range(0, len(neighbour_frequency_indices)),
                        neighbour_frequency_indices,
                        neighbour_direction_indices):

                    delta[index] = (spectral_density[ifreq, idir] - node_value)

                    if partition_label[ifreq, idir] > NOT_ASSIGNED:
                        proximate_partitions_work.append(
                            partition_label[ifreq, idir])

                if numpy.all(delta <= 0) or (partition_label[
                                                 frequency_index, direction_index] > NOT_ASSIGNED):

                    # this is a peak, or already leads to a peak
                    if partition_label[
                        frequency_index, direction_index] == NOT_ASSIGNED:
                        current_label += 1
                        label = current_label
                        proximate_partitions[label] = numpy.array(
                            proximate_partitions_work, dtype='int64')
                    else:
                        label = partition_label[
                            frequency_index, direction_index]
                        proximate_partitions[
                            label] = numpy.append(proximate_partitions[
                                                      label],
                                                  proximate_partitions_work)

                    for i, j in zip(ii, jj):
                        partition_label[i, j] = label

                    break
                else:
                    steepest_index = numpy.argmax(delta)
                    ii.append(neighbour_frequency_indices[steepest_index])
                    jj.append(neighbour_direction_indices[steepest_index])
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

    return partition_label, proximate_partitions


def find_label_closest_partition(label,
                                 proximity: typing.Dict[int, typing.List[int]],
                                 partitions: typing.Dict[int, WaveSpectrum2D]):
    """
    Find the partition whose peak is closest to the partition under considiration
    Only proximate partitions are considered. Basically: find the partition whose
    maximum is closest to the peak of the current partition.

    :param index:
    :param partitions:
    :return:
    """

    # find the labels of the neighbouring partitions.
    proximate_labels = proximity[label]

    if not proximate_labels:
        return None

    # find the wavenumbers associated with the peak
    kx = partitions[label].peak_wavenumber_east()
    ky = partitions[label].peak_wavenumber_north()

    # Initialize the numpy arrays for distance and indices.
    distance = numpy.zeros((len(proximate_labels),))
    indices = numpy.zeros((len(proximate_labels),), dtype='int32')

    for ii, local_label in enumerate(proximate_labels):
        delta_kx = partitions[local_label].peak_wavenumber_east() - kx
        delta_ky = partitions[local_label].peak_wavenumber_north() - ky
        distance[ii] = numpy.sqrt(delta_kx ** 2 + delta_ky ** 2)
        indices[ii] = local_label


    return indices[numpy.argmin(distance)]


def filter_for_low_energy(partitions: typing.Dict[int, WaveSpectrum2D],
                          proximity: typing.Dict[int, typing.List[int]],
                          total_energy, config):

    # If there is only one partition, return
    if len(partitions) == 1:
        return

    for label in list(partitions.keys()):

        if label not in partitions:
            continue

        partition = partitions[label]

        if partition.hm0() == 0:
            closest_label = proximity[label][0]
        else:
            closest_label = find_label_closest_partition(
                label, proximity, partitions)


        if (partition.m0() < config['minimumEnergyFraction'] * total_energy):
            #
            # Merge with the closest partition
            if closest_label:
                merge_partitions(label, closest_label, partitions,
                             proximity)


def merge_partitions(
        source_label,
        target_label,
        partitions: typing.Dict[int, WaveSpectrum2D],
        proximity: typing.Dict[int, typing.List[int]]) -> WaveSpectrum2D:
    """
    Merge one partition into another partition. (**Has side-effects**)
    :param source_index: source index of the partition to merge in the list
    :param target_index: target index of the partition that is merged into
    :param partitions: the list of partitions
    :param partition_label_array: 2D array of labels indicating to what partition
    the cel belongs

    :return: None
    """

    # Get source/target partitions
    partitions[target_label] = partitions[target_label] + partitions[
        source_label]
    partitions.pop(source_label)

    for proximate_labels_to_source_label in proximity[source_label]:
        proximate_to_update = proximity[proximate_labels_to_source_label]
        index = proximate_to_update.index(source_label)

        if proximate_labels_to_source_label == target_label:
            proximity[target_label].pop(index)
            continue

        #
        if proximate_labels_to_source_label not in proximity[target_label]:
            proximity[target_label].append(proximate_labels_to_source_label)

        if target_label in proximate_to_update:
            proximate_to_update.pop(index)
        else:
            proximate_to_update[index] = target_label

    proximity.pop(source_label)
    return partitions[target_label]


def partition_spectrum(spectrum: WaveSpectrum2D, config=None) -> \
        typing.Tuple[typing.Dict[int, WaveSpectrum2D], typing.Dict[
            int, typing.List[int]]]:
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
    density = spectrum.variance_density.copy()
    density[numpy.isnan(density)] = 0.0

    # Get partition descriptions using floodfill. This returns an array that
    # labels for each cell in the spectrum to which partition it belongs, an
    # ordered list with the label as index containing which partitions are
    # adjacent to the current partition and the labels.
    partition_label, proximate = floodfill(
        spectrum.frequency, spectrum.direction * numpy.pi / 180, density)

    # We need to convert back to a list, and we need to do it in a somewhat
    # roundabout way as numba does not support dicts of lists (note, proximate
    # currently is a "numba" dictionarly of numpy arrays)
    proximate_partitions = {}
    for label in proximate:
        proximate_partitions[label] = list(proximate[label])

    # Create Partition objects given the label array from the floodfill.
    partitions = {}
    # Create a dict of all draft partitions
    for label, proximity_list in proximate_partitions.items():
        mask = partition_label == label
        partitions[label] = spectrum.extract(mask,
                                             config['minimumDensityRatio'])
    #
    # # merge low energy partitions
    filter_for_low_energy(partitions, proximate_partitions, spectrum.m0(),
                          config)
    #  if there are multiple sea spectra merge them into a single sea-spectrum
    # merge_sea_spectra(partitions, proximate_partitions, config)

    # Return the partitions and the label array.
    return partitions, proximate_partitions


def partition_marker_array(
        partitions: typing.Dict[int, WaveSpectrum2D]) -> numpy.ndarray:
    a_label = list(partitions.keys())[0]
    marker_array = numpy.zeros(
        partitions[a_label].variance_density.shape) + numpy.nan
    for label, partition in partitions.items():
        mask = ~partition.variance_density.mask
        marker_array[mask] = label
    return marker_array


def sum_partitions(
        partitions: typing.Dict[int, WaveSpectrum2D]) -> WaveSpectrum2D:
    key = list(partitions.keys())[0]
    sum_spec = empty_spectrum2D_like(partitions[key])
    for _, spec in partitions.items():
        sum_spec = spec + sum_spec
    return sum_spec


def is_neighbour(partition_1: WaveSpectrum2D,
                 partition_2: WaveSpectrum2D) -> int:
    footprint = generate_binary_structure(partition_2.variance_density.ndim, 2)

    mask_1 = maximum_filter(partition_1.variance_density.filled(0),
                            footprint=footprint,
                            mode='wrap')
    mask_2 = partition_2.variance_density.filled(0)

    return numpy.nansum(mask_1 * mask_2)
