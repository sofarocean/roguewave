from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D, empty_spectrum2D_like
from .wavespectrum import WaveSpectrum
from .estimators import spec1d_from_spec2d
from .parametric import pierson_moskowitz_frequency
from .partitioning import is_neighbour
from datetime import timedelta
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


class Node():
    def __init__(self,
                 spectrum_index: int,
                 correlation_with_parent: float,
                 spectrum: typing.Union[WaveSpectrum2D, None]):

        self.indices = [spectrum_index]
        self.spectra = [spectrum]
        self.children = []  # type: typing.List["Node"]
        self.correlation_with_previous = [correlation_with_parent]
        self.favored_child = None  # type: typing.Union["Node",None]
        self.parent = None  # type: typing.Union["Node",None]
        self._staged = []

    def stage(self, index, spectrum, correlation):
        self._staged.append((index, spectrum, correlation))

    def commit(self) -> typing.List["Node"]:

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

    def merge(self, indices: typing.List[int],
              spectra: typing.List[WaveSpectrum2D]):
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
                                                    dtype='int32')
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

    def merge_spectra(self, indices: typing.List, spectra):
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


def correlate_spectra_with_nodes(source_spectra: typing.List[WaveSpectrum2D],
                                 nodes: typing.List[Node]) -> numpy.array:
    correlations = numpy.zeros((len(source_spectra), len(nodes)))

    for i_spec, source_spectrum in enumerate(source_spectra):
        for i_node, target_node in enumerate(nodes):
            correlations[i_spec, i_node] = correlate(source_spectrum,
                                                     target_node.spectra[-1])

    return correlations


def create_graph(
        partitions_list: typing.List[typing.Dict[int, WaveSpectrum2D]],
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
                # prune to ensure it is long enough
                # node.prune(min_duration)
                # print(node.duration)
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


def wave_fields_from(root: Node, primary_field=None) -> typing.List[
    typing.List[WaveSpectrum2D]]:
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
