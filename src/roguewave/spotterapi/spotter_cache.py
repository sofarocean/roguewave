"""
Contents: Simple local caching of spotter requests. We simply add a wavefleet
remote resource to a file cache and add custom routines to create and retrieve
data from a cache.

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Classes:
- `WaveFleetResource`, remote resource specification for grabbing data from
   wavefleet

Functions:



Use:

"""

# Import
# =============================================================================


from roguewave.filecache.remote_resources import RemoteResource
from roguewave import filecache
from roguewave import save, load
from roguewave.io.io import NumpyEncoder, object_hook
import json
from typing import List, Callable
from pysofar.spotter import SofarApi


# Constants
# =============================================================================


# Spotter Cache setup.
CACHE_NAME = "spotter_cache"
CACHE_PATH = "~/temporary_roguewave_files/spotter_cache"
CACHE_SIZE_GB = 2


# Classes
# =============================================================================


class WaveFleetResource(RemoteResource):
    """
    Remote resource for downloading data from wavefleet given a "URI".
    The URI in this case is merely a JSON encoded version of the kwargs to
    a call to the appropriate Spotter API download functions. The request type
    and function pointer are registerd as keyword/value pairs in the
    _handlers dictionary.
    """

    URI_PREFIX = "wavefleet://"

    def __init__(self, request_type_handle_mapping: dict, session: SofarApi):
        # Dictionary that maps a request type (str) to a function
        self._handlers = request_type_handle_mapping
        # Sofar session object
        self.session = session

    def download(self) -> Callable[[str, str], bool]:
        # download function conforming to RemoteResource protocol of FileCache.
        def get_data_from_wavefleet(uri: str, filepath: str) -> bool:
            request_type, kwargs = _decode_spotter_cache_uri(uri)
            data = self._handlers[request_type](session=self.session, **kwargs)
            save(data, filepath)
            return True

        return get_data_from_wavefleet

    def handler_defined(self, request_type) -> bool:
        return request_type in self._handlers

    def add_handler(self, request_type, handler):
        if not self.handler_defined(request_type=request_type):
            self._handlers[request_type] = handler
        else:
            raise ValueError(f"A handler is already defined for {request_type}")


# Functions
# =============================================================================
def flush(
    spotter_ids: List[str],
    session: SofarApi,
    handler,
    request_type="get_data",
    **kwargs,
):
    """
    Caching method for the get_data function defined in spotter_api. Note that
    the function handles creation of a cache if not initialized and adding of
    the request handler.

    :param spotter_ids: list of spotter ID's
    :param session: valid sofar API session
    :param handler: Function to execute a given request.
    :param kwargs: Args needed to call the handler.
    :return:
    """

    if not _exists():
        _create_cache(request_type=request_type, handler=handler, session=session)

    if not _handler_is_defined(request_type=request_type):
        return

    uris = [
        _spotter_cache_uri(request_type, spotter_id=_id, **kwargs)
        for _id in spotter_ids
    ]
    filecache.delete_files(uris, cache_name=CACHE_NAME)
    return


def get_data(
    spotter_ids: List[str],
    session: SofarApi,
    handler,
    parallel=True,
    description=None,
    **kwargs,
):
    """
    Caching method for the get_data function defined in spotter_api. Note that
    the function handles creation of a cache if not initialized and adding of
    the request handler.

    :param spotter_ids: list of spotter ID's
    :param session: valid sofar API session
    :param handler: Function to execute a given request.
    :param kwargs: Args needed to call the handler.
    :return:
    """
    REQUEST_TYPE = "get_data"
    if not _exists():
        _create_cache(request_type=REQUEST_TYPE, handler=handler, session=session)

    if not _handler_is_defined(request_type=REQUEST_TYPE):
        _add_handler(request_type=REQUEST_TYPE, handler=handler)

    if description is not None:
        filecache.set("description", description, cache_name=CACHE_NAME)

    filecache.set("parallel", parallel, cache_name=CACHE_NAME)

    uris = [
        _spotter_cache_uri(REQUEST_TYPE, spotter_id=_id, **kwargs)
        for _id in spotter_ids
    ]

    filepaths = filecache.filepaths(uris, cache_name=CACHE_NAME)
    output = [load(filepath) for filepath in filepaths]
    return output


def get_data_search(handler, session: SofarApi, **kwargs):
    """
    Caching method for the search functions defined in spotter_api. Note that
    the function handles creation of a cache if not initialized and adding of
    the request handler.

    :param handler: Function to execute a given request.
    :param session: valid sofar API session
    :param kwargs: Args needed to call the handler.

    :return:
    """
    REQUEST_TYPE = "search"
    if not _exists():
        _create_cache(request_type=REQUEST_TYPE, handler=handler, session=session)

    if not _handler_is_defined(request_type=REQUEST_TYPE):
        _add_handler(request_type=REQUEST_TYPE, handler=handler)

    uri = _spotter_cache_uri(REQUEST_TYPE, **kwargs)
    filepath = filecache.filepaths(uri, cache_name=CACHE_NAME)[0]
    return load(filepath)


# Module internal functions
# =============================================================================


def _spotter_cache_uri(request_type: str, **kwargs):
    """
    Encode a set of keyword arguments as a wavefleet "URI". The wavefleet uri
    is a string that has the form:

        wavefleet://[request_type]/[json_serialized_kwargs]

    To note; this is only used to create a hashable string that uniquely
    specifies the request.

    :param request_type: request type unique identifier that has a
        corresponding handling function in the WaveFleetResource
    :param kwargs: json serializabe dict. See NumpyEncoder for custom additions
    :return:  uri
    """
    kwargs["request_type"] = request_type
    return "wavefleet://" + json.dumps(kwargs, cls=NumpyEncoder)


def _decode_spotter_cache_uri(uri: str):
    """
    Decode a "wavefleet uri" back into its kwargs form together with
    the request type. A "wavefleet uri" is a string that has the form:

        wavefleet://[request_type]/[json_serialized_kwargs]

    To note; this is only used to create a hashable string that uniquely
    specifies the request.

    :param uri: wavefleet "URI"
    :return:  request_type and kwargs that serve as input to the
        request_type handler.
    """
    uri = uri.replace("wavefleet://", "")
    kwargs = json.loads(uri, object_hook=object_hook)
    request_type = kwargs.pop("request_type")
    return request_type, kwargs


def _create_cache(request_type, handler, session):
    if not _exists():
        wavefleetresource = WaveFleetResource({request_type: handler}, session)
        filecache.create_cache(
            CACHE_NAME,
            cache_path=CACHE_PATH,
            resources=[wavefleetresource],
            cache_size_GB=CACHE_SIZE_GB,
        )


def _exists() -> bool:
    """
    Check if a spotter cache already exists
    :return: True or False
    """
    return filecache.exists(CACHE_NAME)


def _add_handler(request_type, handler):
    """
    Add a handler to the WaveFleetResource object of the spotter file cache
    that handles the given request type. If a handler with the given
    request type already exists we do _NOT_ update.

    :param request_type: request type unique identifier
    :param handler: function that handles kwargs associated with the request
        type
    :return:
    """

    cache = filecache.get_cache(CACHE_NAME)
    # Remote resources are stored in the resource list of the cache object.
    for remote_resource in cache.resources:
        # We need to get the WaveFleetResource Object to add the handler too
        if isinstance(remote_resource, WaveFleetResource):
            remote_resource.add_handler(request_type, handler)
            break
    else:
        # Note that we require that the wavefleet object is there. If the
        # cache was created through create_cache this should be the case.
        raise ValueError("No valid WaveFleetResource object instantiated")


def _handler_is_defined(request_type):
    """
    Check if a handler is defined for the request type.
    :param request_type: request type unique identifier.
    :return:
    """
    cache = filecache.get_cache(CACHE_NAME)
    for remote_resource in cache.resources:
        if isinstance(remote_resource, WaveFleetResource):
            return remote_resource.handler_defined(request_type)
    else:
        # Note that we require that the wavefleet object is there. If the
        # cache was created through create_cache this should be the case.
        raise ValueError("No valid WaveFleetResource object instantiated")
