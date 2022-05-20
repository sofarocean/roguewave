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

from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput, \
    empty_spectrum2D_like
from .operators import spectrum1D_time_filter, spectrum2D_time_filter
from .wavespectrum import WaveSpectrum, BulkVariables
from .parametric import pierson_moskowitz_frequency
from pandas import DataFrame
import numpy
import scipy.ndimage
import typing
from .estimators import spect2d_from_spec1d
import numba

default_partition_config = {
    'minimumEnergyFraction': 0.01,
    'minimumRelativeDistance': 0.1,
}


class SeaSwellData(typing.TypedDict):
    """

    """
    sea: DataFrame
    total_swell: DataFrame
    partitions_swell: typing.List[DataFrame]


class Partition():
    def __init__(self, source_spectrum: WaveSpectrum2D, partition_labels,
                 proximate_partitions, label):
        self.proximate_partitions = proximate_partitions
        self.mask = partition_labels == label
        self.label = label

        partition_density = numpy.zeros_like(source_spectrum.variance_density)
        partition_density[self.mask] = source_spectrum.variance_density[
            self.mask]

        self.spectrum = WaveSpectrum2D(
            WaveSpectrum2DInput(frequency=source_spectrum.frequency,
                                varianceDensity=partition_density,
                                timestamp=source_spectrum.timestamp,
                                latitude=source_spectrum.latitude,
                                longitude=source_spectrum.longitude,
                                directions=source_spectrum.direction))
        self._update()

    def merge(self, density, proximate_regions, mask, label):
        self.spectrum.variance_density = self.spectrum.variance_density + density
        self.spectrum._update()
        self.mask = self.mask | mask
        self.proximate_partitions = \
            numpy.unique(numpy.concatenate(
                (self.proximate_partitions, proximate_regions)))
        mask = self.proximate_partitions != label
        self.proximate_partitions = self.proximate_partitions[mask]

    def _update(self):
        self.peak_wavenumber = self.spectrum.peak_wavenumber()
        self.direction = self.spectrum.peak_direction()
        self.peak_wavenumber_east = numpy.cos(
            self.direction * numpy.pi / 180) * self.peak_wavenumber
        self.peak_wavenumber_north = numpy.sin(
            self.direction * numpy.pi / 180) * self.peak_wavenumber
        self.m0 = self.spectrum.m0()
        self.tm01 = self.spectrum.peak_period()
        self.directional_width = self.spectrum.bulk_spread()

    def is_sea_partition(self) -> bool:
        """
        Identify whether or not it is a sea partion. Use 1D method for 2D
        spectra in section 3 of:

        Portilla, J., Ocampo-Torres, F. J., & Monbaliu, J. (2009).
        Spectral partitioning and identification of wind sea and swell.
        Journal of atmospheric and oceanic technology, 26(1), 107-122.

        :return: boolean indicting it is sea
        """
        portilla = True
        spec1D = self.spectrum.e
        peak_index = numpy.argmax(spec1D)
        peak_frequency = self.spectrum.frequency[peak_index]
        if portilla:
            density = spec1D[peak_index]
            return pierson_moskowitz_frequency(peak_frequency, peak_frequency
                                               ) <= density
        else:
            freq = self.spectrum.frequency
            pm = pierson_moskowitz_frequency(freq, peak_frequency)
            index = numpy.nonzero(pm < self.spectrum.e)[0]

            if len(index) == 0:
                return False

            factor = numpy.mean(
                self.spectrum.e[index[0]:] / pm[index[0]:]) > 0.8
            m0_tail = self.spectrum.m0(fmin=freq[index[0]])
            m0_peak = self.spectrum.m0(fmax=freq[index[0]])
            energy_fraction = m0_tail / m0_peak > 0.5

            return factor & energy_fraction

    def is_swell_partition(self):
        return not self.is_sea_partition()

numba.njit()
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

numba.njit()
def floodfill(frequency: numpy.ndarray, direction: numpy.ndarray,
              spectral_density: numpy.ndarray, partition_label: numpy.ndarray):
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

    def distance(freq1, angle1, freq2, angle2, grav=9.81):
        k1 = (freq1 * numpy.pi) ** 2 / grav
        k2 = (freq2 * numpy.pi) ** 2 / grav

        return numpy.sqrt(
            (k1 * numpy.cos(angle1) - k2 * numpy.cos(angle2)) ** 2 +
            (k1 * numpy.sin(angle1) - k2 * numpy.sin(angle2)) ** 2
        )

    def jacobian(frequency_hertz,grav=9.81):
        return 1/(8*numpy.pi**2) * grav/frequency_hertz

    number_of_directions = spectral_density.shape[1]
    number_of_frequencies = spectral_density.shape[0]
    proximate_partitions = []

    current_label = NOT_ASSIGNED

    proximate_partitions = []

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

                delta_k = distance(frequency[neighbour_frequency_indices],
                                   direction[neighbour_direction_indices],
                                   frequency[frequency_index],
                                   direction[direction_index])
                node_value = spectral_density[frequency_index, direction_index]
                delta = (spectral_density[
                             neighbour_frequency_indices, neighbour_direction_indices] - node_value) / delta_k

                neighbour_assigned_mask = partition_label[
                                              neighbour_frequency_indices, neighbour_direction_indices] > NOT_ASSIGNED
                neighbour_assigned_labels = partition_label[
                    neighbour_frequency_indices[neighbour_assigned_mask],
                    neighbour_direction_indices[neighbour_assigned_mask]]
                proximate_partitions_work += [ii for ii in neighbour_assigned_labels]

                if numpy.all(delta <= 0) or (partition_label[
                                                frequency_index, direction_index] > NOT_ASSIGNED):
                    # this is a peak, or already leads to a peak
                    ii = numpy.array(ii, dtype='int64')
                    jj = numpy.array(jj, dtype='int64')

                    if partition_label[
                        frequency_index, direction_index] == NOT_ASSIGNED:
                        current_label += 1
                        label = current_label
                        proximate_partitions.append(proximate_partitions_work)
                    else:
                        label = partition_label[
                            frequency_index, direction_index]
                        proximate_partitions[
                            label] += proximate_partitions_work
                    partition_label[ii, jj] = label

                    break

                else:
                    steepest_index = numpy.argmax(delta)
                    ii.append(neighbour_frequency_indices[steepest_index])
                    jj.append(neighbour_direction_indices[steepest_index])
                    continue

        #
    for ii in range(0,len(proximate_partitions)):
        proximate_partitions[ii] = numpy.unique(proximate_partitions[ii])
    return proximate_partitions


def find_peaks(density: numpy.ndarray):
    """
    Find the peaks of the frequency-direction spectrum.
    :param density: 2D variance density
    :return: List of indices indicating the maximum
    """

    density[0:5, :] = 0.0

    # create a search region
    neighborhood = scipy.ndimage.generate_binary_structure(density.ndim, 2)

    # Set the local neighbourhood to the the maximum value, use wrap.
    # Note that technically frequency wrapping is incorrect- but unlikely to
    # cause issues. First we apply the maximum filter .with 9 point footprint
    # to set each pixel of the output to the local maximum
    filtered = scipy.ndimage.maximum_filter(
        density, footprint=neighborhood
    )

    # Then we find maxima based on equality with input array
    maximum_mask = filtered == density

    # Remove possible background (0's)
    background_mask = density == 0

    # Remove peaks in background
    maximum_mask[background_mask] = False

    # return indices of maxima
    ii, jj = numpy.where(maximum_mask)

    # sort from lowest maximum value to largest maximum value.
    sorted = numpy.flip(numpy.argsort(density[ii, jj]))

    return ii[sorted], jj[sorted]

numba.njit()
def find_partitions(frequency, direction, density):
    #
    density = density.copy()
    #
    # peak_frequencty_indices, peak_direction_indices = find_peaks(density)
    partition_label = numpy.zeros_like(density, dtype='int64') + NOT_ASSIGNED

    proximate_partitions_adjacency_list = []


    proximate_partitions_adjacency_list = floodfill(frequency, direction,
                                                    density, partition_label)

    number_of_partitions = len(proximate_partitions_adjacency_list)
    for ii in range(0, number_of_partitions):
        proximate_partition_indices = proximate_partitions_adjacency_list[ii]
        for proximate_partition_index in proximate_partition_indices:
            proximate_partitions_adjacency_list[proximate_partition_index] = (
                numpy.append(
                    proximate_partitions_adjacency_list[
                        proximate_partition_index],
                    ii)
            )

    for ii in range(0, number_of_partitions):
        proximate_partitions_adjacency_list[ii] = numpy.unique(
            proximate_partitions_adjacency_list[ii])

    return partition_label, proximate_partitions_adjacency_list, numpy.arange(
        0, number_of_partitions)


def label_to_index_mapping(partitions):
    labels = numpy.array([x.label for x in partitions])
    label_to_index = {}
    for index, label in enumerate(labels):
        label_to_index[label] = index
    return label_to_index


def find_index_closest_partition(index, partitions: typing.List[Partition]):
    """
    Find the partition whose peak is closest to the partition under considiration
    Only proximate partitions are considered. Basically: find the partition whose
    maximum is closest to the peak of the current partition.

    :param index:
    :param partitions:
    :return:
    """

    # Map the label used to mark the the partition to the ordinal numeral in
    # the list. E.g.: the partition with Label=3 might be the first in our list
    # of partitions (i.e partitions[0].label=3).
    label_to_index = label_to_index_mapping(partitions)

    # find the labels of the neighbouring partitions.
    proximate_labels = partitions[index].proximate_partitions

    # find the wavenumbers associated with the peak
    kx = partitions[index].peak_wavenumber_east
    ky = partitions[index].peak_wavenumber_north
    k = partitions[index].peak_wavenumber

    # Make sure that we do not consider the "self" label as proximate.
    mask = proximate_labels != partitions[index].label
    proximate_labels = proximate_labels[mask]

    # Initialize the numpy arrays for distance and indices.
    distance = numpy.zeros((len(proximate_labels),))
    indices = numpy.zeros((len(proximate_labels),), dtype='int32')

    for ii, label in enumerate(proximate_labels):
        proximate_index = label_to_index[label]

        delta_kx = partitions[proximate_index].peak_wavenumber_east - kx
        delta_ky = partitions[proximate_index].peak_wavenumber_north - ky
        distance[ii] = numpy.sqrt(delta_kx ** 2 + delta_ky ** 2) / k
        indices[ii] = proximate_index

    minimum_index = numpy.argmin(distance)
    return indices[minimum_index], distance[minimum_index]


def filter_for_low_energy(partitions: typing.List[Partition], total_energy,
                          partition_label_array, config):
    index = 0
    while True:
        partition = partitions[index]
        if len(partitions) == 1:
            break

        closest_index, distance = find_index_closest_partition(index,
                                                               partitions)

        if ((partition.m0 < config['minimumEnergyFraction'] * total_energy) or
                (distance < config['minimumRelativeDistance'])
        ):
            #
            # Merge with the closest partition
            merge_partitions(index, closest_index, partitions,
                             partition_label_array)

            # pop
            partitions.pop(index)

        else:
            index += 1

        if index >= len(partitions):
            break


def merge_sea_spectra(partitions: typing.List[Partition],
                      partition_label_array, config):
    sea_partitions = []
    for index, partition in enumerate(partitions):
        if partition.is_sea_partition():
            sea_partitions.append(index)

    if len(sea_partitions) > 1:
        for index in sea_partitions[1:]:
            merge_partitions(index, sea_partitions[0], partitions,
                             partition_label_array)

        for index in sorted(sea_partitions[1:], reverse=True):
            del partitions[index]



def merge_partitions(source_index: int,
                     target_index: int,
                     partitions: typing.List[Partition],
                     partition_label_array: numpy.ndarray
                     ) -> None:
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
    source = partitions[source_index]
    target = partitions[target_index]

    # Merge source into target
    target.merge(
        source.spectrum.variance_density,
        source.proximate_partitions,
        source.mask,
        source.label
    )

    # update all the labels so that they now point to the new partition
    label_to_index = label_to_index_mapping(partitions)

    # Update the list of proximate partitions of the source to indicate they
    # are now proximate to the target
    for proximate_partition_label in source.proximate_partitions:
        if proximate_partition_label == source.label:
            continue
        proximate_index = label_to_index[proximate_partition_label]
        proximate_partition = partitions[proximate_index]
        mask = proximate_partition.proximate_partitions == source.label
        proximate_partition.proximate_partitions[mask] = target.label

        # Make sure the list remains unique (if the target was already proximate)
        proximate_partition.proximate_partitions = numpy.unique(
            proximate_partition.proximate_partitions)

    # update the labels in the array
    partition_label_array[source.mask] = target.label

    return None


def partition_spectrum(spectrum: WaveSpectrum2D, config=None) -> typing.Tuple[
    typing.List[Partition], numpy.ndarray]:
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
    partition_label_array, proximate_partitions, labels = find_partitions(
        spectrum.frequency, spectrum.direction * numpy.pi/180, density)

    # Create Partition objects given the label array from the floodfill.
    partitions = []

    # Create a list of all draft partitions
    for proximate_region, label in zip(proximate_partitions, labels):
        partitions.append(
            Partition(spectrum, partition_label_array, proximate_region,
                      label))

    # merge low energy partitions
    filter_for_low_energy(partitions, spectrum.m0(), partition_label_array,
                          config)

    #  if there are multiple sea spectra merge them into a single sea-spectrum
    merge_sea_spectra(partitions, partition_label_array,
                      config)

    # Return the partitions and the label array.
    return partitions, partition_label_array


class BulkPartitionVariables():
    def __init__(self, partitions: typing.List[Partition]):
        self._partitions_swell = []

        total_swell_spectrum = empty_spectrum2D_like(partitions[0].spectrum)
        self._sea = total_swell_spectrum.bulk_variables()

        for partition in partitions:
            spectrum = partition.spectrum
            if partition.is_sea_partition():
                self._sea = spectrum.bulk_variables()
            else:
                self._partitions_swell.append(spectrum.bulk_variables())
                total_swell_spectrum.variance_density = (
                        total_swell_spectrum.variance_density +
                        spectrum.variance_density)
        total_swell_spectrum._update()
        self._total_swell = total_swell_spectrum.bulk_variables()

        if self._sea.hm0 == 0:
            self._sea._nanify()

        if not self._partitions_swell:
            self._partitions_swell = [self._total_swell]

    @property
    def sea(self) -> BulkVariables:
        return self._sea

    @property
    def total_swell(self) -> BulkVariables:
        return self._total_swell

    @property
    def partitions_swell(self) -> typing.List[BulkVariables]:
        return self._partitions_swell

    @property
    def number_of_swells(self):
        return len(self._partitions_swell)

    def fill_to(self, n):
        number_of_swells = self.number_of_swells
        if number_of_swells == n:
            return

        if number_of_swells > n:
            self._partitions_swell = self._partitions_swell[:n]
            return

        bulk = BulkVariables(None)
        bulk.timestamp = self.total_swell.timestamp
        bulk.latitude = self.total_swell.latitude
        bulk.longitude = self.total_swell.longitude
        for ii in range(number_of_swells, n):
            self._partitions_swell.append(bulk)


def _gen_dataframe_from_partitions(
        bulk: typing.List[BulkVariables]) -> DataFrame:
    """
    From a partition, create a dataframe.
    :param bulk:
    :return:
    """
    n = len(bulk)
    data = {}
    properties = ['m0', 'hm0', 'tm01', 'tm02', 'peak_period', 'peak_direction',
                  'peak_spread',
                  'bulk_direction', 'bulk_spread', 'peak_frequency',
                  'peak_wavenumber', 'latitude', 'longitude', 'timestamp']

    for key in properties:
        if key == 'timestamp':
            data[key] = numpy.empty((n,), dtype=object)
        else:
            data[key] = numpy.zeros((n,))

    for i, b in enumerate(bulk):
        for key in properties:
            data[key][i] = getattr(b, key)

    return DataFrame.from_dict(data)


def _homogonize_bulk_swell(bulk: typing.List[BulkPartitionVariables]) -> \
        typing.List[BulkPartitionVariables]:
    """
    Ensure that all entries in the list have the same number of swell partitions
    :param bulk:
    :return:
    """
    max_partitions = numpy.array([b.number_of_swells for b in bulk]).max()
    for b in bulk:
        b.fill_to(max_partitions)
    return bulk


def sea_swell_data(spectra: typing.List[WaveSpectrum],
                   time_filter=True, verbose=True,
                   filter_window=None) -> SeaSwellData:
    """

    :param spectra: list of wavespectra objects.
    :return: dictionary (typed: SeaSwellData) that contains:
        { "sea": pandas dataframe
          "total_swell": pandas dataframe
          "partitions_swell": list[pandas dataframe]

    """
    bulk = []
    # For each partition do:

    if time_filter:
        if isinstance(spectra[0], WaveSpectrum1D):
            spectra: typing.List[WaveSpectrum1D]
            spectra = spectrum1D_time_filter(spectra, filter_window)
        elif isinstance(spectra[0], WaveSpectrum2D):
            spectra: typing.List[WaveSpectrum2D]
            spectra = spectrum2D_time_filter(spectra, filter_window)

    for ii, spectrum in enumerate(spectra):

        if verbose:
            print(ii)

        if isinstance(spectrum, WaveSpectrum1D):
            spec2d = spect2d_from_spec1d(spectrum)
        elif isinstance(spectrum, WaveSpectrum2D):
            spec2d = spectrum
        else:
            raise Exception('Unknown spectral type')

        # Partition the spectrum
        partition, _ = partition_spectrum(spec2d)

        # calculate bulk variables for each partition
        bulk.append(BulkPartitionVariables(partition))

    sea = _gen_dataframe_from_partitions([b.sea for b in bulk])
    total_swell = _gen_dataframe_from_partitions([b.total_swell for b in bulk])

    #
    # Create a list of bulk descriptions of the swell, and make sure that all
    # bulk descriptions contain the same number of swells. This makes processing
    # easier (no other reason).
    #
    bulk = _homogonize_bulk_swell(bulk)

    # now all data have the same number of swell partitions
    max_partitions = bulk[0].number_of_swells

    # For each partition, create the dataframe.
    partitions_swell = []
    for ii in range(0, max_partitions):
        partitions_swell.append(
            _gen_dataframe_from_partitions(
                [b.partitions_swell[ii] for b in bulk])
        )

    # Return Data
    return SeaSwellData(sea=sea, total_swell=total_swell,
                        partitions_swell=partitions_swell)
