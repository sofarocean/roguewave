from .spectrum1D import WaveSpectrum1D, WaveSpectrum1DInput
from .spectrum2D import WaveSpectrum2D, WaveSpectrum2DInput, \
    empty_spectrum2D_like
from .wavespectrum import WaveSpectrum, BulkVariables
from .parametric import pierson_moskowitz_frequency
from pandas import DataFrame
import numpy
import scipy.ndimage
import typing
from . import spect2d_from_spec1d

"""
This module implements a partitioning algorith for use on 2D spectra.

# Usage


# Implementation

"""

NOT_ASSIGNED = -1

default_partition_config = {
    'minimumEnergyFraction': 0.01,
    'minimumRelativeDistance': 0.1,
}


class SeaSwellData(typing.TypedDict):
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


def neighbours(peak_direction_index, peak_frequency_index,
               number_of_directions, number_of_frequencies, diagonals=True):
    # First we need to calculate the next/prev angles in frequency ann
    # direction space, noting that direction space is periodic.
    next_direction_index = (peak_direction_index + 1) % number_of_directions
    prev_direction_index = (peak_direction_index - 1) % number_of_directions
    prev_frequency_index = peak_frequency_index - 1
    next_frequency_index = peak_frequency_index + 1

    # Add the
    ii = [peak_frequency_index, peak_frequency_index]
    jj = [prev_direction_index, next_direction_index]
    if diagonals:
        if peak_frequency_index > 0:
            ii += [prev_frequency_index, prev_frequency_index,
                   prev_frequency_index]
            jj += [prev_direction_index, peak_direction_index,
                   next_direction_index]
        if peak_frequency_index < number_of_frequencies - 1:
            ii += [next_frequency_index, next_frequency_index,
                   next_frequency_index]
            jj += [prev_direction_index, peak_direction_index,
                   next_direction_index]
    else:
        if peak_frequency_index > 0:
            ii += [prev_frequency_index]
            jj += [peak_direction_index]
        if peak_frequency_index < number_of_frequencies - 1:
            ii += [next_frequency_index]
            jj += [peak_direction_index]

    return numpy.array(ii), numpy.array(jj)


def floodfill(spectral_density: numpy.ndarray, partition_label: numpy.ndarray,
              peak_frequency_index: int, peak_direction_index: int):
    """
    Flood fill algorithm. We try to find the region that belongs to a peak according
    to inverse waterfall.

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

    label = partition_label.max() + 1
    number_of_directions = spectral_density.shape[1]
    number_of_frequencies = spectral_density.shape[0]

    queue = [(peak_frequency_index, peak_direction_index)]
    in_queue = numpy.zeros_like(spectral_density, dtype='int64')
    in_queue[peak_frequency_index, peak_direction_index] = 1
    proximate_partitions = []

    while True:
        # 1. Pop the next node to consider
        peak_frequency_index, peak_direction_index = queue.pop()

        # 2. Add to current partition ("fill")
        partition_label[peak_frequency_index, peak_direction_index] = label

        # 3. Find neighbours to flood next:
        # 3.1 Create neighbour list:
        neighbour_frequency_index, neighbour_direction_index = neighbours(
            peak_direction_index, peak_frequency_index,
            number_of_directions, number_of_frequencies)

        # 3.2 Find neighbours to consider
        node_value = spectral_density[
            peak_frequency_index, peak_direction_index]
        neighbour_values = spectral_density[
            neighbour_frequency_index, neighbour_direction_index]

        # Find proximate partitions and add them to the list
        neighbour_labels = partition_label[
            neighbour_frequency_index, neighbour_direction_index]

        proximate_partitions += list(
            neighbour_labels[neighbour_labels > NOT_ASSIGNED])

        # From all neighbours add if
        to_add = (
            # Neighbours are smaller
                (neighbour_values < node_value * 1.01) &
                # are not in queue
                (in_queue[
                     neighbour_frequency_index, neighbour_direction_index] == 0) &
                # are not already assigned a partition
                (partition_label[
                     neighbour_frequency_index, neighbour_direction_index] == NOT_ASSIGNED)
        )
        # get indices of neighbours to add
        neighbour_frequency_index_to_add = neighbour_frequency_index[to_add]
        neighbour_direction_index_to_add = neighbour_direction_index[to_add]

        # 4 Add neighbours found to queue
        queue += [(ix, iy) for ix, iy in
                  zip(neighbour_frequency_index_to_add,
                      neighbour_direction_index_to_add)]
        in_queue[
            neighbour_frequency_index_to_add, neighbour_direction_index_to_add] = 1

        # 5. If queue is empty we are done
        if len(queue) <= 0:
            break
        #

    return numpy.unique(proximate_partitions)


def find_peaks(density: numpy.ndarray):
    """
    Find the peaks of the frequency-direction spectrum.
    :param density: 2D variance density
    :return: List of indices indicating the maximum
    """

    density[0:5, :] = 0.0

    # create a search region
    neighborhood = scipy.ndimage.generate_binary_structure(density.ndim, 2)

    # Find local maxima, use wrap. Note that technically frequency wrapping is
    # incorrect- but unlikely to cause issues. First we apply the maximum filter
    # with 9 point footprint to set each pixel of the output to the local maximum
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

    sorted = numpy.flip(numpy.argsort(density[ii, jj]))

    return ii[sorted], jj[sorted]


def find_partitions(density):
    #
    density = density.copy()
    #
    peak_frequencty_indices, peak_direction_indices = find_peaks(density)
    partition_label = numpy.zeros_like(density, dtype='int64') - 1

    proximate_partitions_adjacency_list = []
    number_of_partitions = 0

    for peak_frequency_index, peak_direction_index in zip(
            peak_frequencty_indices, peak_direction_indices):

        if partition_label[
            peak_frequency_index, peak_direction_index] > NOT_ASSIGNED:
            continue

        number_of_partitions += 1
        #
        # Do floodfill on partition label (note- partition_label gets changed)
        proximate_partitions_adjacency_list += [
            floodfill(density, partition_label, peak_frequency_index,
                      peak_direction_index)]

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
    label_to_index = label_to_index_mapping(partitions)
    proximate_labels = partitions[index].proximate_partitions

    kx = partitions[index].peak_wavenumber_east
    ky = partitions[index].peak_wavenumber_north
    k = partitions[index].peak_wavenumber

    mask = proximate_labels != partitions[index].label
    proximate_labels = proximate_labels[mask]

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
            partitions.pop(index)


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
        density)

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
                   time_smooth=True) -> SeaSwellData:
    """

    :param spectra: list of wavespectra objects.
    :return: dictionary (typed: SeaSwellData) that contains:
        { "sea": pandas dataframe
          "total_swell": pandas dataframe
          "partitions_swell": list[pandas dataframe]

    """

    def get_spec2D(spectrum: WaveSpectrum) -> WaveSpectrum2D:
        # If 1D spectrum- make 2D using mem estimate for directions.
        if isinstance(spectrum, WaveSpectrum1D):
            return spect2d_from_spec1d(spectrum)
        elif isinstance(spectrum, WaveSpectrum2D):
            return spectrum
        else:
            raise Exception('Unknown spectral format')

    bulk = []
    # For each partition do:

    for ii, spectrum in enumerate(spectra):
        print(ii)
        if time_smooth:
            density = (0.25 * spectra[max(ii - 1, 0)].variance_density +
                       0.5 * spectra[ii].variance_density +
                       0.25 * spectra[
                           min(ii + 1, len(spectra) - 1)].variance_density
                       )
            a1 = (0.25 * spectra[max(ii - 1, 0)].a1 +
                  0.5 * spectra[ii].a1 +
                  0.25 * spectra[min(ii + 1, len(spectra) - 1)].a1
                  )
            b1 = (0.25 * spectra[max(ii - 1, 0)].b1 +
                  0.5 * spectra[ii].b1 +
                  0.25 * spectra[min(ii + 1, len(spectra) - 1)].b1
                  )
            a2 = (0.25 * spectra[max(ii - 1, 0)].a2 +
                  0.5 * spectra[ii].a2 +
                  0.25 * spectra[min(ii + 1, len(spectra) - 1)].a2
                  )
            b2 = (0.25 * spectra[max(ii - 1, 0)].b2 +
                  0.5 * spectra[ii].b2 +
                  0.25 * spectra[min(ii + 1, len(spectra) - 1)].b2
                  )
            spec1d = WaveSpectrum1D(WaveSpectrum1DInput(frequency=
                                                 spectrum.frequency, varianceDensity=density,
                                                 timestamp=spectrum.timestamp,
                                                 latitude=spectrum.latitude,
                                                 longitude=spectrum.longitude, a1=a1, b1=b1,
                                                 a2=a2, b2=b2))
            spec2d = spect2d_from_spec1d(spec1d)
        else:
            spec2d = spect2d_from_spec1d(spectrum)

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
