"""
Contents: Simple file caching routines to interact with a file cache.

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Functions:
- `filepaths`, given URI's return a filepath to the locally stored
   version
- `exists`, does a cache with a given name exists
- `create_cache`, create a cache with a given name and custom properties.
- `delete_cache`, delete files associated with the cache.
- `delete_default`, delete files associated with the default cache.
- `delete_files`, remove entries from a given cache.
- `_get_cache`, get Cache object corresponding to the name (for internal use
   only)
"""

# Import
# =============================================================================
import os
from typing import List, Tuple, Union, Dict, Iterable, Callable
from .cache_object import TEMPORARY_DIRECTORY, CACHE_SIZE_GB, FileCache
from .remote_resources import RemoteResource

# Constants
# =============================================================================


DEFAULT_CACHE_NAME = "__default__"

# Private Module Variables
# =============================================================================


# This dictionary contains all instantiated FileCache objects as values and
# the object name as key.
_ACTIVE_FILE_CACHES = {}  # type: Dict[str,FileCache]


# Main public function.
# =============================================================================
def set(name, value, cache_name: str = None):
    cache = get_cache(cache_name)
    setattr(cache, name, value)


def filepaths(
    uris: Union[List[str], str],
    cache_name: str = None,
) -> Union[List[str], Tuple[List[str], List[bool]]]:
    """
    Return the full file path to locally stored objects corresponding to the
    given URI

    :param uris: List of uris, or a single uri
    :param cache_name: name of the cache to use. If None, a default cache will
    be initialized automatically (if not initialized) and used.
    :param return_cache_hits: return whether or not the files were already in
        cache or downloaded from the remote source (cache hit or miss).

    :return: List Absolute paths to the locally stored versions corresponding
        to the list of URI's. IF return_cache_hits=True, additionally return
        a list of cache hits as the second entry of the return tuple.
    """
    return get_cache(cache_name)[uris]


def remove_directive_function(directive: str, name: str, cache_name=None):
    _ = get_cache(cache_name).remove_directive_function(directive, name)


def set_directive_function(
    directive: str,
    name: str,
    post_process_function: Union[Callable[[str], None], Callable[[str], bool]] = None,
    cache_name=None,
):
    #
    _ = get_cache(cache_name).set_directive_function(
        directive, name, post_process_function
    )


def exists(cache_name: str):
    """
    Check if the cache name already exists
    :param cache_name: name for the cache to be created. This name is used
            to retrieve files from the cache.
    :return: True if exists, False otherwise
    """
    return cache_name in _ACTIVE_FILE_CACHES


def create_cache(
    cache_name: str,
    cache_path: str = TEMPORARY_DIRECTORY,
    cache_size_GB: Union[int, float] = CACHE_SIZE_GB,
    do_cache_eviction_on_startup: bool = False,
    download_in_parallel=True,
    resources: List[RemoteResource] = None,
) -> None:
    """
    Create a file cache. Created caches *must* have unique names and
    cache_paths.

    :param cache_name: name for the cache to be created. This name is used
            to retrieve files from the cache.
    :param cache_path: path to store cache. If path does not exist it will be
            created.
    :param cache_size_GB:  Maximum size of the cache in GiB. If cache exceeds
            the size, then files with oldest access/modified dates get deleted
            until everthing fits in the cache again. Fractional values (floats)
            are allowed.
    :param do_cache_eviction_on_startup: do_cache_eviction_on_startup: whether
            we ensure the cache size conforms to the given size on startup.
            If set to true, a cache directory that exceeds the maximum size
            will be reduced to max size. Set to False by default in which case
            an error occurs. The latter to prevent eroneously evicting files
            from a cache that was previously created on purpose with a larger
            size that the limit.
    :param download_in_parallel: Download in paralel from resource. Per default 10
            worker threads are created.

    :return:
    """
    cache_path = os.path.abspath(os.path.expanduser(cache_path))

    if cache_name in _ACTIVE_FILE_CACHES:
        raise ValueError(f"Cache with name {cache_name} is already initialized")

    for key, cache in _ACTIVE_FILE_CACHES.items():
        if cache.path == cache_path:
            raise ValueError(
                f"Error when creating cache with name: "
                f'"{cache_name}". \n A cache named: "{key}" '
                f"already uses the path {cache_path} "
                f"for caching.\n "
                f"Multiple caches cannot share the same path."
            )

    _ACTIVE_FILE_CACHES[cache_name] = FileCache(
        cache_path,
        size_GB=cache_size_GB,
        do_cache_eviction_on_startup=do_cache_eviction_on_startup,
        parallel=download_in_parallel,
        resources=resources,
    )
    return


def delete_cache(cache_name):
    """
    Delete all files associated with a cache and remove cache from available
    caches. To note: all files are deleted, but the folder itself is not.

    :param cache_name: Name of the cache to be deleted
    :return:
    """
    if not exists(cache_name):
        raise ValueError(f"Cache with name {cache_name} does not exist")

    cache = _ACTIVE_FILE_CACHES.pop(cache_name)
    cache.purge()


def delete_default():
    """
    Clean up the default cache.

    :return:
    """
    if exists(DEFAULT_CACHE_NAME):
        delete_cache(DEFAULT_CACHE_NAME)


def delete_files(uris: Union[str, Iterable[str]], cache_name: str) -> None:
    """
    Remove given key(s) from the cache
    :param uris: list of keys to remove
    :param cache_name: name of initialized cache.
    :return:
    """
    if not isinstance(uris, Iterable) or isinstance(uris, str):
        uris = [uris]

    cache = get_cache(cache_name)
    for key in uris:
        cache.remove(key)


def get_cache(cache_name: str) -> FileCache:
    """
    Get a valid cache object, error if the name does not exist.
    :param cache_name: Name of the cache
    :return: Cache object
    """

    if cache_name is None:
        cache_name = DEFAULT_CACHE_NAME

    if not exists(cache_name):
        if cache_name == DEFAULT_CACHE_NAME:
            create_cache(cache_name)
        else:
            raise ValueError(f"Cache with name {cache_name} does not exist.")

    return _ACTIVE_FILE_CACHES[cache_name]
