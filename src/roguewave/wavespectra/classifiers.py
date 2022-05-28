from .spectrum1D import WaveSpectrum1D
from .spectrum2D import WaveSpectrum2D, empty_spectrum2D_like
from .wavespectrum import WaveSpectrum
from .estimators import spec1d_from_spec2d
from .parametric import pierson_moskowitz_frequency
from .partitioning import merge_partitions, find_label_closest_partition
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
                numpy.nansum(source * target)
            # numpy.corrcoef(source.flatten(), target.flatten(), )[0, 1]
        else:
            correlation[ii] = 0

    i = numpy.nanargmax(correlation)
    return labels[i], correlation[i]


def find_all_matching_partitions(
        source_spectra: typing.Dict[int, WaveSpectrum2D],
        target_spectra: typing.Dict[int, WaveSpectrum2D]):
    correlation = {}

    for label, source_spectrum in source_spectra.items():
        correlation[label] = find_matching_partition(source_spectrum,
                                                     target_spectra)
    return correlation


def link_partitions(prev: typing.Dict[int, WaveSpectrum2D],
                    curr: typing.Dict[int, WaveSpectrum2D],
                    proximity,
                    is_first=False,
                    threshold=0.1):
    curr_to_prev = find_all_matching_partitions(curr, prev)

    labels = list(curr.keys())

    start = {}
    paths = {}

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\t correlation between current, previous and next")
        logger.debug(f"\t\t strength prev curr next strength")
        for label in labels:
            prev_label, strength_prev = curr_to_prev[label]
            logger.debug(
                f"\t\t{strength_prev:4.2e} {prev_label:04d} {label:04d} "
            )

    logger.debug(f"\t Looping over connections")
    for label in labels:
        prev_label, strength_prev = curr_to_prev[label]
        prev_in_start = prev_label in start

        # Another path with the same start point already exists
        if prev_in_start:
            # if the current path has a stronger correlation, replace
            logger.debug(
                f"\t\t label {prev_label} is already connected to {start[prev_label]['label']}")
            if strength_prev > start[prev_label]['correlation']:
                logger.debug(
                    f"\t\t  + new connection {label} has higher correlation")
                if start[prev_label][
                    'correlation'] < threshold * strength_prev:
                    logger.debug(
                        f"\t\t  + the old connection is too small {threshold}")
                    target = find_label_closest_partition(
                        start[prev_label]['label'],
                        proximity, curr)
                    merge_partitions(start[prev_label]['label'], target, curr,
                                     proximity)
                    paths.pop(start[prev_label]['label'])
                else:
                    logger.debug(
                        f"\t\t  + the old connection is a new partition {threshold}")
                    paths[start[prev_label]['label']][0] = - start[prev_label][
                        'label']

                start[prev_label] = {'label': label,
                                     'correlation': strength_prev}
            else:
                logger.debug(
                    f"\t\t  + old connection {label} has higher correlation")
                if threshold * start[prev_label][
                    'correlation'] > strength_prev:
                    logger.debug(
                        f"\t\t  + the new connection is too small")

                    target = find_label_closest_partition(label,
                                                          proximity, curr)
                    merge_partitions(label, target, curr, proximity)
                    continue
                else:
                    logger.debug(
                        f"\t\t  + the new connection is a new partition")
                    prev_label = - label
        else:
            logger.debug(
                f"\t\t label {prev_label} is a new connection for {label}")
            start[prev_label] = {'label': label, 'correlation': strength_prev}

        paths[label] = [prev_label, strength_prev]

    # # Step two- check for the threshold
    # labels = list(paths.keys())
    # for label in labels:
    #     path = paths[label]
    #     if path[0] is not None:
    #         if start[path[0]]['correlation'] < threshold:
    #             path[0] = None

    return paths


class Node():

    def __init__(self, spectrum_index, correlation_with_parent,
                 spectrum: typing.Union[WaveSpectrum2D, None]):
        self.start_index = spectrum_index
        self.end_index = spectrum_index
        self.index = [spectrum_index]
        self.spectra = [spectrum]
        self.children = [] # type: typing.List["Node"]
        self.correlation_with_previous = [correlation_with_parent]
        self.favored_child = None
        self.parent = None # type: typing.Union["Node",None]
        self._staged = []

    def stage(self, index, spectrum, correlation):
        self._staged.append( (index, spectrum, correlation) )

    def commit(self):
        if not self._staged:
            return []

        if len( self._staged ) > 1:
            max_cor = 0
            favored_child = -1
            for index, _,  correlation in self._staged:
                if correlation > max_cor:
                    max_cor = correlation
                    favored_child = index


            for index, spectrum, correlation in self._staged:
                child = Node(index,
                             correlation,
                             spectrum)
                self.add_child(child)
                if index == favored_child:
                    self.favored_child = child

            self._staged = []
            return self.children
        else:
            self.add_spectrum( *self._staged[0])
            self._staged = []
            return [self]

    def add_spectrum( self,index, spectrum, correlation ):
        self.end_index = index
        self.spectra.append(spectrum)
        self.correlation_with_previous.append(correlation)
        self.index.append(index)

    def add_child(self, child: "Node"):
        #
        child.parent = self
        self.children.append(child)

    @property
    def duration(self):
        return self.spectra[-1].timestamp - self.spectra[0].timestamp

    def prune(self, min_duration:timedelta):
        if self.duration > min_duration:
            return
        else:
            self.parent.remove_child(self)

    def remove_child(self, child:"Node"):
        for index, sibling in enumerate(self.children):
            if sibling is child:
                child_index = index
                break
        else:
            raise Exception('not a child')
        self.children.pop(child_index)

        corr = 0
        favored_child = None
        for candidate_child in self.children:
            if candidate_child.correlation_with_previous[0] >= corr:
                corr = candidate_child.correlation_with_previous[0]
                favored_child = candidate_child

        if not favored_child:
            print('eh')
        self.favored_child = favored_child

        assert not child.children
        self.merge( child.index, child.spectra )


    def merge(self, indices, spectra):
        if not len(indices):
            return

        if not self.children:
            self.merge_spectra(indices,spectra)
        else:
            if indices[0] <= self.end_index:
                istart = self.index.index(indices[0])
                length = len(self.index) - istart
            else:
                length = 0

            self.merge_spectra(indices[0:length],spectra[0:length])
            if not length >= len(self.index):
                self.favored_child.merge(indices[length:], spectra[length:])

            # Since correlation between children and parents has changed, update
            # the correlations.
            corr = 0
            favored_child = None
            for candidate_child in self.children:
                if candidate_child.correlation_with_previous[0] >= corr:
                    corr = candidate_child.correlation_with_previous[0]
                    favored_child = candidate_child
            self.favored_child = favored_child


    def merge_spectra(self,indices:typing.List, spectra):
        if not indices:
            return

        if indices[0] > self.end_index:
            assert indices[0] == self.end_index + 1
            istart = self.end_index + 1
        else:
            istart = self.index.index(indices[0])

        jj = istart
        for index,spectrum in zip(indices[istart:],spectra[istart:]):
            if index > self.index[-1]:
                correlation = correlate(self.spectra[-1],spectrum)
                self.add_spectrum( index, spectrum, correlation )
            else:
                self.spectra[jj] = self.spectra[jj] + spectrum
                if jj == 0:
                    correlation = correlate( self.parent.spectra[-1], self.spectra[jj])
                else:
                    correlation = correlate(self.spectra[jj-1],
                                            self.spectra[jj])
                self.correlation_with_previous[jj] = correlation


def correlate( source:WaveSpectrum2D,target:WaveSpectrum2D ):
    source = source.variance_density.filled(0).flatten()
    target = target.variance_density.filled(0).flatten()
    return numpy.nansum(source * target)

def correlate_spectra_with_nodes(source_spectra: typing.List[WaveSpectrum2D],
                                 nodes: typing.List[Node]) -> numpy.array:
    correlations = numpy.zeros((len(source_spectra), len(nodes)))

    for i_spec, source_spectrum in enumerate(source_spectra):
        for i_node, target_node in enumerate(nodes):

            correlations[i_spec, i_node] = correlate(source_spectrum,target_node.spectra[-1])

    return correlations


def create_graph(
        partitions_list: typing.List[typing.Dict[int, WaveSpectrum2D]], min_duration):
    root = Node(-1, 0, None )
    for label, spectrum in partitions_list[0].items():
        empty = empty_spectrum2D_like( spectrum )
        root.add_child(Node(0, 1,spectrum))
    root.spectra[0] = empty

    active_nodes = root.children
    for index in range(1, len(partitions_list)):

        new_active_set = []
        labels = list(partitions_list[index].keys())
        partitions = [ partitions_list[index][label] for label in labels ]

        correlations = correlate_spectra_with_nodes(partitions, active_nodes)
        new_to_previous = numpy.argmax(correlations, axis=1)
        for new_index, max_index in enumerate(new_to_previous):
            active_nodes[max_index].stage(
                index,
                partitions[new_index],
                correlations[new_index,max_index]
            )
        for node in active_nodes:
            if nodes := node.commit():
                new_active_set += nodes
            else:
                # No spectra have been added - this is the end for this partition
                # prune to ensure it is long enough
                node.prune(min_duration)
        active_nodes = new_active_set


    for node in active_nodes:
        node.prune(min_duration)


    return root


def wave_fields_from(root:Node, primary_field=None)->typing.List[typing.List[WaveSpectrum2D]]:
    fields = []
    if root.parent is None:
        for child in root.children:
            fields += wave_fields_from(child,[])
    else:
        for spectrum in root.spectra:
            primary_field.append( spectrum )

        if root.children:
            for child in root.children:
                if child is root.favored_child:
                    fields += wave_fields_from(child,primary_field)
                else:
                    fields += wave_fields_from(child,[])
        else:
            fields += [primary_field]

    return fields