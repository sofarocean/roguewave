import numpy
from datetime import timedelta
from typing import List, Union, Dict, overload
from pandas import DataFrame
from roguewave import WaveSpectrum2D
from roguewave.wavespectra import logger
from roguewave.wavespectra.spectrum2D import empty_spectrum2D_like
from roguewave.wavespectra.wavespectrum import WaveSpectrum, \
    extract_bulk_parameter
from .partitioning import is_neighbour, default_partition_config, partition_spectrum
from roguewave.tools import _print
from .classifiers import is_sea_spectrum
from multiprocessing import cpu_count, get_context

DEFAULT_CONFIG_PARTITION_SPECTRA = {
    'partitionConfig': default_partition_config,
    'parallel': False, # DOES NOT WORK
    'fieldFiltersettings': {
        'filter': True,
        'maxDeltaPeriod': 2,
        'maxDeltaDirection': 20
    }
}


class Node():
    def __init__(self,
                 spectrum_index: int,
                 correlation_with_parent: float,
                 spectrum: Union[WaveSpectrum2D, None]):

        self.indices = [spectrum_index]
        self.spectra = [spectrum]
        self.children = []  # type: List["Node"]
        self.correlation_with_previous = [correlation_with_parent]
        self.favored_child = None  # type: Union["Node",None]
        self.parent = None  # type: Union["Node",None]
        self._staged = []

    def stage(self, index, spectrum, correlation):
        self._staged.append((index, spectrum, correlation))

    def commit(self) -> List["Node"]:

        # if no partitions are stages- nothing to be done, return empty list
        # (signifying no nodes were added)
        if not self._staged:
            return []

        # Partitions are staged...
        if len(self._staged) > 1:
            # if multiple partitions are staged- this is a branch point, add
            # children
            for index, spectrum, correlation in self._staged:
                child = Node(index,
                             correlation,
                             spectrum)
                self.add_child(child)

            self._staged = []

            # we return the children- these are now active to continue the path
            return self.children
        else:
            #
            # Single partition, just add to the current spectra
            self.add_spectrum(*self._staged[0])
            self._staged = []

            # return self- is still active.
            return [self]

    def add_spectrum(self, index, spectrum, correlation):
        self.spectra.append(spectrum)
        self.correlation_with_previous.append(correlation)
        self.indices.append(index)

    def add_child(self, child: "Node"):
        #
        child.parent = self
        self.children.append(child)
        self.update_favored_child()

    @property
    def duration(self):
        delta = self.spectra[-1].timestamp - self.spectra[0].timestamp
        # if self.parent is None:
        #     return delta
        #
        # if self.parent.favored_child is self:
        #     delta = delta + self.parent.duration
        return delta

    def prune(self, min_duration: timedelta):
        if self.duration > min_duration:
            return False
        else:
            self.parent.remove_child(self)
            return True

    def update_favored_child(self):
        self.favored_child = None
        corr = 0
        for candidate_child in self.children:
            if candidate_child.correlation_with_previous[0] >= corr:
                corr = candidate_child.correlation_with_previous[0]
                self.favored_child = candidate_child

    def remove_child(self, child: "Node"):
        assert not child.children
        for index, sibling in enumerate(self.children):
            if sibling is child:
                child_index = index
                break
        else:
            raise Exception('not a child')

        self.children.pop(child_index)
        self.merge(child.indices, child.spectra)
        child.parent = None

    def merge(self, indices: List[int],
              spectra: List[WaveSpectrum2D]):
        if not len(indices):
            return

        if not self.children:
            self.merge_spectra(indices, spectra)
        else:
            if indices[0] <= self.indices[-1]:
                istart = self.indices.index(indices[0])
                length = min(len(self.indices) - istart, len(indices))

            else:
                length = 0
                assert indices[0] == self.indices[-1] + 1

            self.merge_spectra(indices[0:length], spectra[0:length])
            if length == len(spectra):
                return

            number_of_neighbourpoints = numpy.zeros(len(self.children),
                                                    dtype='float64')
            for ii, child in enumerate(self.children):
                    number_of_neighbourpoints[ii] = is_neighbour(spectra[length],
                                                             child.spectra[0])

            index_of_child_to_merge_into = numpy.argmax(
                number_of_neighbourpoints)

            self.children[index_of_child_to_merge_into].merge(indices[length:],
                                                              spectra[length:])

        # Since correlation between children and parents has changed, update
        # the correlations.
        self.update_favored_child()

    def merge_spectra(self, indices: List, spectra):
        if not indices:
            return

        if indices[0] > self.indices[-1]:
            assert indices[0] == self.indices[-1] + 1
            istart = self.indices[-1] + 1
        else:
            istart = self.indices.index(indices[0])

        jj = istart
        for index, spectrum in zip(indices, spectra):
            #
            if index > self.indices[-1]:
                correlation = correlate(self.spectra[-1], spectrum)
                self.add_spectrum(index, spectrum, correlation)
            else:
                self.spectra[jj] = self.spectra[jj] + spectrum
                if jj == 0:
                    correlation = correlate(self.parent.spectra[-1],
                                            self.spectra[jj])
                else:
                    correlation = correlate(self.spectra[jj - 1],
                                            self.spectra[jj])
                self.correlation_with_previous[jj] = correlation
            jj = jj + 1


def correlate(source: WaveSpectrum2D, target: WaveSpectrum2D):
    source = source.variance_density.filled(0).flatten()
    target = target.variance_density.filled(0).flatten()
    return numpy.nansum(source * target)


def correlate_spectra_with_nodes(source_spectra: List[WaveSpectrum2D],
                                 nodes: List[Node]) -> numpy.array:
    correlations = numpy.zeros((len(source_spectra), len(nodes)))

    for i_spec, source_spectrum in enumerate(source_spectra):
        for i_node, target_node in enumerate(nodes):
            correlations[i_spec, i_node] = correlate(source_spectrum,
                                                     target_node.spectra[-1])

    return correlations


def create_graph(
        partitions_list: List[Dict[int, WaveSpectrum2D]],
        min_duration):
    label = list(partitions_list[0].keys())[0]
    empty = empty_spectrum2D_like(partitions_list[0][label])
    root = Node(-1, 0, empty)
    for label, spectrum in partitions_list[0].items():
        root.add_child(Node(0, 1, spectrum))

    active_nodes = root.children
    for index in range(1, len(partitions_list)):

        new_active_set = []
        labels = list(partitions_list[index].keys())
        partitions = [partitions_list[index][label] for label in labels]

        # Correlate all active nodes with new candidate partitions
        correlations = correlate_spectra_with_nodes(partitions, active_nodes)

        # get the strongest correlations between new partitions and previous
        # nodes
        new_to_previous = numpy.argmax(correlations, axis=1)

        # connect new partition to previous nodes. We first stage the commits
        # so we now how many partitions are connected to a previous node
        for new_index, max_index in enumerate(new_to_previous):
            active_nodes[max_index].stage(
                index,
                partitions[new_index],
                correlations[new_index, max_index]
            )

        for node in active_nodes:
            if nodes := node.commit():
                new_active_set += nodes
            else:
                # No spectra have been added - this is the end for this partition
                pass
        active_nodes = new_active_set

    tree_prune(root, min_duration)
    return root


def tree_prune(root: Node, min_duration: timedelta):
    """
    Remove branches from the tree that contain only a few speectra.
    :param root:
    :param min_duration:
    :return:
    """

    iteration = 0
    logger.debug(f"Pruning the tree")
    while branch_prune(root, min_duration):
        iteration += 1
        logger.debug(f"\t Iteration {iteration:04d}")
    return


def branch_prune(root: Node, min_duration: timedelta) -> bool:
    pruned = False
    if root.children:
        for child in root.children:
            pruned = pruned or branch_prune(child, min_duration)
    else:
        if root.parent:
            pruned = root.prune(min_duration)
    return pruned


def wave_fields_from(root: Node, primary_field=None) -> List[
    List[WaveSpectrum2D]]:
    """
    Create different swell/sea fields based on a given connectivity tree.

    :param root: root of the tree structure
    :param primary_field: memory parameter in the recursive construction.
    :return:
    """

    # Initialize the output fields of this root to eempty
    fields = []

    # If we do not have a parent we are the root of three- each of the children
    # is considered a valid field>
    if root.parent is None:
        for child in root.children:
            # Recurisively call method on children. Out output field from children
            # to current field
            fields += wave_fields_from(child, [])
    else:
        # add spectra of the current node to the wave field
        for spectrum in root.spectra:
            primary_field.append(spectrum)

        # if we have children, add their spectra too
        if root.children:
            for child in root.children:
                if child is root.favored_child:
                    # if this is the "favored" child this continuous the primary partition
                    fields += wave_fields_from(child, primary_field)
                else:
                    # if this is not the favored child, this is a new field.
                    fields += wave_fields_from(child, [])
        else:
            # if no children, this is the end. Return the primary partition as
            # a field.
            fields += [primary_field]

    return fields


def filter_field(field:List[WaveSpectrum2D], min_duration:timedelta, max_delta_period,max_delta_direction):

    new_fields = [ [] ] # type: List[List[WaveSpectrum2D]]
    current_field = 0
    for index, spec in enumerate(field):
        if index == 0:
            new_fields[current_field].append(spec)
            continue

        previous_field = field[index-1]
        delta_period = numpy.abs(  spec.tm01() - previous_field.tm01()  )
        delta_direction = numpy.abs( (spec.bulk_direction() - previous_field.bulk_direction() +180 )%360 - 180)

        if delta_direction > max_delta_direction or delta_period > max_delta_period:
            current_field+=1
            new_fields.append([])
        new_fields[current_field].append(spec)

    to_drop = []
    for index,new_field in enumerate(new_fields):
        if new_field[-1].timestamp - new_field[0].timestamp < min_duration:
            to_drop.append(index)

    for index in sorted(to_drop,reverse=True):
        new_fields.pop(index)
    return new_fields


def filter_fields(fields:List[List[WaveSpectrum2D]], min_duration:timedelta, max_delta_period=2,max_delta_direction=20):
    new_fields = []
    for field in fields:
        new_fields += filter_field(field, min_duration,max_delta_period,max_delta_direction)
    return new_fields


def bulk_parameters_partitions( partitions:List[List[WaveSpectrum2D]] )->List[DataFrame]:
    bulk = []
    for label,partition in enumerate(partitions):
        df = DataFrame()
        for variable in WaveSpectrum.bulk_properties:
            df[variable] = extract_bulk_parameter(variable, partition)
        df['sea'] = numpy.array(is_sea_spectrum(partition))
        bulk.append(df)
    return bulk

def worker(spectrum):
    return partition_spectrum(spectrum)

def partition_spectra(spectra2D: List[WaveSpectrum2D],
                      minimum_duration: timedelta,
                      config=None, verbose=False) -> List[List[WaveSpectrum2D]]:
    if config:
        for key in config:
            assert key in DEFAULT_CONFIG_PARTITION_SPECTRA, f"{key} is not a valid configuration entry"

        config = DEFAULT_CONFIG_PARTITION_SPECTRA | config
    else:
        config = DEFAULT_CONFIG_PARTITION_SPECTRA


    # Step 1: Partition the data
    _print(verbose, ' - Partition Data')
    raw_partitions = []
    if config['parallel']:
        with get_context("spawn").Pool(processes=cpu_count()) as pool:
            output = pool.map(worker, spectra2D, chunksize=len(spectra2D)//cpu_count() )
        raw_partitions = [ partition for partition, _ in output ]
    else:
        for index, spectrum in enumerate(spectra2D):
            _print(verbose, f'\t {index:05d} out of {len(spectra2D)}')
            partitions, _ = partition_spectrum(spectrum, config['partitionConfig'])
            raw_partitions.append(partitions)

    # Step 2: create a graph
    _print(verbose, ' - Create Graph')
    graph = create_graph(raw_partitions, minimum_duration)

    # Step 3: create wave field from the graph
    _print(verbose, ' - Create Wave Fields From Graph')
    wave_fields = wave_fields_from(graph)

    # Step 4: Postprocessing

    # Apply a filter on the bulk parameters
    _print(verbose, ' - Apply Bulk Filter')
    if config['fieldFiltersettings']['filter']:
        wave_fields = filter_fields(
            wave_fields,
            min_duration=minimum_duration,
            max_delta_period=config['fieldFiltersettings']['maxDeltaPeriod'],
            max_delta_direction=config['fieldFiltersettings'][
                'maxDeltaDirection']
        )
    _print(verbose, '*** Done ***\n' + 80 * '-' + '\n\n')
    return wave_fields

# -----------------------------------------------------------------------------
#                       Boilerplate Interfaces
# -----------------------------------------------------------------------------
@overload
def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: Dict[str, List[List[WaveSpectrum2D]]]
) -> Dict[str, List[DataFrame]]: ...

@overload
def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: List[List[WaveSpectrum2D]]
) -> List[DataFrame]: ...


def get_bulk_partitions_from_spectral_partitions(
        spectral_partitions: Union[Dict[str, List[List[WaveSpectrum2D]]], List[
            List[WaveSpectrum2D]]]) -> Union[Dict[
                                                 str, List[DataFrame]], List[
                                                 DataFrame]]:
    if isinstance(spectral_partitions, dict):
        output = {}
        for key in spectral_partitions:
            output[key] = bulk_parameters_partitions(spectral_partitions[key])

    elif isinstance(spectral_partitions, list):
        output = bulk_parameters_partitions(spectral_partitions)
    else:
        raise Exception('Cannot process input')

    return output


def get_spectral_partitions_from_2dspectra(
        spectra: Union[Dict[str, List[WaveSpectrum2D]], List[WaveSpectrum2D]],
        minimum_duration: timedelta,
        config=None,
        verbose=False) -> Union[
    Dict[str, List[List[WaveSpectrum2D]]], List[List[WaveSpectrum2D]]]:
    #

    if isinstance(spectra, dict):
        output = {}
        for key, item in spectra.items():
            output[key] = partition_spectra(item,
                                            minimum_duration,
                                            config, verbose)
    elif isinstance(spectra, list):
        output = partition_spectra(spectra,
                                   minimum_duration,
                                   config,
                                   verbose)
    else:
        raise Exception('Cannot process input')

    return output