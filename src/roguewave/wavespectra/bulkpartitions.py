import typing

import numpy
from pandas import DataFrame

from roguewave import WaveSpectrum1D, WaveSpectrum2D
from roguewave.wavespectra import spec2d_from_spec1d
from roguewave.wavespectra.operators import spectrum1D_time_filter, \
    spectrum2D_time_filter
from roguewave.wavespectra.partitioning import Partition, \
    partition_spectrum, merge_partitions, find_index_closest_partition
from roguewave.wavespectra.spectrum2D import empty_spectrum2D_like
from roguewave.wavespectra.wavespectrum import BulkVariables, WaveSpectrum


class SeaSwellData(typing.TypedDict):
    """

    """
    sea: DataFrame
    total_swell: DataFrame
    partitions: typing.List[DataFrame]


class BulkPartitionVariables():
    def __init__(self, partitions: typing.List[Partition]):
        # total_swell_spectrum = empty_spectrum2D_like(partitions[0].spectrum)
        self._raw_partitions = partitions

    @property
    def sea(self) -> BulkVariables:
        for x in self._raw_partitions:
            if x.is_sea_partition():
                return x.spectrum.bulk_variables()
        else:
            bulk = BulkVariables(None)
            bulk.timestamp = self._raw_partitions[0].spectrum.timestamp
            bulk.latitude = self._raw_partitions[0].spectrum.latitude
            bulk.longitude = self._raw_partitions[0].spectrum.longitude
            return bulk

    @property
    def total_swell(self) -> BulkVariables:
        total_swell_spectrum = empty_spectrum2D_like(
            self._raw_partitions[0].spectrum)

        ii = 0
        for x in self._raw_partitions:
            if not x:
                continue

            if x.is_swell_partition():
                ii += 1
                total_swell_spectrum.variance_density = total_swell_spectrum.variance_density + x.spectrum.variance_density

        if ii > 0:
            total_swell_spectrum._update()
            return total_swell_spectrum.bulk_variables()
        else:
            bulk = BulkVariables(None)
            bulk.timestamp = self._raw_partitions[0].spectrum.timestamp
            bulk.latitude = self._raw_partitions[0].spectrum.latitude
            bulk.longitude = self._raw_partitions[0].spectrum.longitude
            return bulk

    @property
    def partitions(self) -> typing.List[BulkVariables]:
        out = []

        for x in self._raw_partitions:
            if x:
                bulk = x.spectrum.bulk_variables()
            else:
                bulk = BulkVariables(None)
                bulk.timestamp = self.total_swell.timestamp
                bulk.latitude = self.total_swell.latitude
                bulk.longitude = self.total_swell.longitude
            out.append(bulk)

        return out

    @property
    def number_of_partitions(self):
        return len(self._raw_partitions)

    def renumber(self, mapping: typing.List[int]):

        assert len(mapping) == self.number_of_partitions
        new = []
        for new_index, old_index in enumerate(mapping):
            new.append(self._raw_partitions[old_index])
        self._partitions_swell = new

    def match_exists(self, partition:Partition):
        return self.find_matching_partition(partition) >= 0

    def find_matching_partition(self, partition:Partition,pr=False):

        correlation = numpy.zeros(len(self._raw_partitions))

        source = partition.spectrum.variance_density[:] - numpy.mean(partition.spectrum.variance_density[:])
        source =source.flatten()
        for ii, part in enumerate(self._raw_partitions):
            if part is not None:
                target = part.spectrum.variance_density[:] - numpy.mean(part.spectrum.variance_density[:])
                target=target.flatten()
                correlation[ii] = numpy.corrcoef( source.flatten(), target.flatten(), )[0,1]
            else:
                correlation[ii] = 0

        i = numpy.nanargmax(correlation)

        if pr:
            #print(correlation)
            pass
        if correlation[i] >= 0.7:
            return i
        else:
            return -1

    def fill_to(self, n):
        number_of_partitions = len(self._raw_partitions)
        if number_of_partitions == n:
            return

        if number_of_partitions > n:
            self._raw_partitions = self._raw_partitions[:n]
            return

        for ii in range(number_of_partitions, n):
            self._raw_partitions.append(None)

    def closest_index(self,source_index):
        target_index, _ = find_index_closest_partition(source_index,
                                                    self._raw_partitions)
        return target_index

    def merge(self, source_index , target_index):
        merge_partitions(source_index, target_index, self._raw_partitions)
        return


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
    max_partitions = numpy.array([b.number_of_partitions for b in bulk]).max()
    for b in bulk:
        b.fill_to(max_partitions)
    return bulk


def match_partitions(bulk_partitions: typing.List[BulkPartitionVariables]) -> \
typing.List[BulkPartitionVariables]:
    for index, partition in enumerate(bulk_partitions):
        if index == 0:
            continue

        mapping = numpy.zeros((partition.number_of_partitions,),
                              dtype='int32') - 1
        previous = bulk_partitions[index - 1]

        mapped = []
        notmapped = []
        empty = []
        for old_index, swell in enumerate(partition.partitions):
            if numpy.isnan(swell.hm0):
                empty.append(old_index)
            else:
                index = previous.find_matching_partition(
                    partition._raw_partitions[old_index])
                if index >= 0:
                    mapped.append((old_index, index))
                else:
                    notmapped.append(old_index)

        ii = 0
        for map in mapped:
            mapping[map[1]] = map[0]

        jj = -1
        if notmapped:
            for index, map in enumerate(mapping):
                if map < 0:
                    jj += 1
                    mapping[index] = notmapped[jj]
                    if jj == len(notmapped)-1:
                        break

        jj = -1
        if empty:
            for index, map in enumerate(mapping):
                if map < 0:
                    jj += 1
                    mapping[index] = empty[jj]
        partition.renumber(mapping)

    for index, bulk in enumerate(bulk_partitions):

        iu = index + 1
        id = index - 1
        if index == len(bulk_partitions) - 1:
            iu = id

        if index == 0:
            id = iu

        prev = bulk_partitions[id]
        next = bulk_partitions[iu]
        #

        to_merge = {}
        if bulk.number_of_partitions <= 1:
            continue

        for ipart, partition in enumerate(bulk.partitions):
            #
            if numpy.isnan(partition.hm0):
                continue

            match_in_prev = prev.match_exists(bulk._raw_partitions[ipart])
            match_in_next = next.match_exists(bulk._raw_partitions[ipart])

            if index == 3:
                print('---')
                print(index,ipart, match_in_prev, match_in_next)
                print(index,ipart, prev.find_matching_partition(bulk._raw_partitions[ipart],pr=True) )
                print(index, ipart, next.find_matching_partition(
                    bulk._raw_partitions[ipart],pr=True))


            if not (  match_in_prev or match_in_next ):
                target = bulk.closest_index(ipart)
                if target in to_merge:
                    target = to_merge[target]

                if not index==target:
                    to_merge[ipart] = target

        for source,target in to_merge.items():
            bulk.merge(source,target)

        to_merge = [ x for x in to_merge]
        for index in sorted(to_merge, reverse=True):
            del bulk._raw_partitions[index]

    return bulk_partitions


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
            spec2d = spec2d_from_spec1d(spectrum)
        elif isinstance(spectrum, WaveSpectrum2D):
            spec2d = spectrum
        else:
            raise Exception('Unknown spectral type')

        # Partition the spectrum
        partition, _ = partition_spectrum(spec2d)

        # calculate bulk variables for each partition
        bulk.append(BulkPartitionVariables(partition))

    bulk = _homogonize_bulk_swell(bulk)
    bulk = match_partitions(bulk)

    sea = _gen_dataframe_from_partitions([b.sea for b in bulk])
    total_swell = _gen_dataframe_from_partitions([b.total_swell for b in bulk])

    #
    # Create a list of bulk descriptions of the swell, and make sure that all
    # bulk descriptions contain the same number of swells. This makes processing
    # easier (no other reason).
    #
    bulk = _homogonize_bulk_swell(bulk)

    # now all data have the same number of swell partitions
    max_partitions = bulk[0].number_of_partitions

    # For each partition, create the dataframe.
    partitions_swell = []
    for ii in range(0, max_partitions):
        partitions_swell.append(
            _gen_dataframe_from_partitions(
                [b.partitions[ii] for b in bulk])
        )

    # Return Data
    return SeaSwellData(sea=sea, total_swell=total_swell,
                        partitions=partitions_swell)
