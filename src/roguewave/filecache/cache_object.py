"""
Contents: Simple file caching routines that automatically cache remote files
          locally for use.

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Classes:
- `FileCache`, main class implementing the Caching structure. Should not
   directly be invoked. Instead, fetching/cache creation is controlled by a
   set of function defined below

Functions:

"""
import hashlib
import os
from _warnings import warn
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union, List, Tuple, Callable
from tqdm import tqdm
from roguewave import logger
from .remote_resources import RemoteResourceS3, \
    RemoteResourceHTTPS, RemoteResource, RemoteResourceLocal
from .exceptions import _RemoteResourceUriNotFound

TEMPORARY_DIRECTORY = '~/temporary_roguewave_files/filecache/'
CACHE_SIZE_GB = 5
MAXIMUM_NUMBER_OF_WORKERS = 10
KILOBYTE = 1000
MEGABYTE = 1000 * KILOBYTE
GIGABYTE = 1000 * MEGABYTE


class FileCache:
    """
    Simple file caching class that when given an URI locally stores the
    file in the cache directory and returns the path to the file. The file
    remains in storage until the cache directory exceeds a prescribed size,
    in which case files with oldest access/modified dates get deleted first
    until everything fits in the cache again. Any time a file is accessed it's
    modified date gets updated so that often used files automaticall remain in
    cache.

    The files are stored locally in the directory specified on class
    initialization, as:

        [path]/CACHE_PREFIX + md5_hash_of_URI + CACHE_POSTFIX

    The pre- and post fix are added so we have an easy pattern to distinguish
    cache files from other files.

    Methods
      * __getitem__(keys) : accept a simgle uri_key or a list of URI's and
        returns filepaths to local copies thereof. You would typically use the
            cache[keys] notation instead of the dunder method.
      * purge() clear all contents of the cache (destructive, deletes all local
        files).

    Usage:

        cache = FileCache()
        list_of_local_file_names = cache[ [list of URI's ] ]

    # do stuff with file
    ...
    """

    CACHE_FILE_PREFIX = 'cachefile_'
    CACHE_FILE_POSTFIX = '_cachefile'

    def __init__(self,
                 path: str = TEMPORARY_DIRECTORY,
                 size_GB: Union[float, int] = CACHE_SIZE_GB,
                 do_cache_eviction_on_startup: bool = False,
                 resources:List[RemoteResource]=None,
                 parallel=True,
                 allow_for_missing_files=False,
                 post_process_function: Callable[[str],None] = None,
                 validate_function: Callable[[str], None] = None,
                 ):
        """
        Initialize Cache
        :param path: path to store cache. If path does not exist it will be
            created.
        :param size_GB: Maximum size of the cache in GiB. If cache exceeds
            the size, then files with oldest access/modified dates get deleted
            until everthing fits in the cache again. Fractional values (floats)
            are allowed.
        :param do_cache_eviction_on_startup: whether we ensure the cache size
            conforms to the given size on startup. If set to true, a cache
            directory that exceeds the maximum size will be reduced to max
            size. Set to False by default in which case an error occurs. The
            latter to prevent eroneously evicting files from a cache that was
            previously created on purpose with a larger size that the limit.


        """

        self.path = os.path.expanduser(path)
        self.max_size = int(size_GB * GIGABYTE)
        self.parallel = parallel
        self.allow_for_missing_files = allow_for_missing_files

        # create the path if it does not exist
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Some counters to keep track of total cache misses, hits and
        # evictions. No downstream use right now/
        self._cache_misses = 0
        self._cache_hits = 0
        self._cache_evictions = 0

        # initialize the cache.
        self._entries = {}  # the key/value pair cache
        self._initialize_cache(do_cache_eviction_on_startup)

        # Post processing and validation functions
        self.post_process_function = post_process_function
        self.validate_function = validate_function

        # download resources
        if resources is None:
            self.resources = [RemoteResourceS3(),
                              RemoteResourceHTTPS(),
                              RemoteResourceLocal()]
        else:
            self.resources = resources

    def _post_process(self,filepath:str):
        """
        Call custom post processing function that gets called after first
        download- if set.
        :param filepaths: list of filepaths
        :return: none
        """
        if self.post_process_function is None:
            return
        else:
            self.post_process_function(filepath)

    def _validate(self,filepath:str)->bool:
        """
        Call custom post processing function that gets called after first
        download- if set.
        :param filepaths: list of filepaths
        :return: none
        """
        if self.validate_function is None:
            return True
        else:
            return self.validate_function(filepath)

    def _cache_file_name(self, uri: str) -> str:
        """
        Return the filename that corresponds to the given uri. We construct
        the file name using a simple md5 hash of the uri string prefixed
        with a cache file prefix. THe later is introduced to seperate cache
        files in a path from user files (and avoid including/deleting those).

        :param uri: valis uri"
        :return: valid cache file
        """
        return self.CACHE_FILE_PREFIX + _hashname(
            uri) + self.CACHE_FILE_POSTFIX

    def _cache_file_path(self, uri: str) -> str:
        """
        Construct the path where the given uri is stored locally.

        :param uri: valis uri of form "bucket/key"
        :return: valid cache file
        """
        return os.path.join(self.path, self._cache_file_name(uri))

    def _get_cache_files(self) -> List[str]:
        """
        Find all files that are currently a member of the cache.
        :return:
        """
        _cache_files = []
        for path, dirs, files in os.walk(self.path):
            # Return all files that are "cache" objects. This is a safety if
            # other user files are present, so that these don't accidentally
            # evicted from the cache (aka deleted).
            return [file for file in files if
                    file.startswith(self.CACHE_FILE_PREFIX) and
                    file.endswith(self.CACHE_FILE_POSTFIX)]
        else:
            return []

    def _initialize_cache(self, do_cache_eviction_on_startup: bool) -> None:
        """
        Initialize the file cache. Look on disk for files in the cache path
        that have the required prefix and load these into the cache. Once
        loaded, we do a check whether or not the cache is full and if we need
        to remove files.
        :param do_cache_eviction_on_startup: see description under __init__
        :return:
        """
        self._entries = {}
        for file in self._get_cache_files():
            filepath = os.path.join(self.path, file)
            self._entries[file] = filepath

        # See if cache is still < size
        if do_cache_eviction_on_startup:
            self._cache_eviction()
        else:
            if self._size() > self.max_size:
                raise ValueError('The cache currently existing on disk '
                                 'exceeds the maximum cache size of the '
                                 'current cache.\n Either increase the cache '
                                 'size of the object or allow eviction of '
                                 'files on startup.')

    def in_cache(self, uris) -> List[bool]:
        # make sure input is a list
        if isinstance(uris, str):
            uris = [uris]

        # Create the hashes from the URI's
        hashes = [self._cache_file_name(uri) for uri in uris]
        return [self._is_in_cache(_hash) for _hash in hashes]

    def _is_in_cache(self, _hash: str) -> bool:
        """
        Check if a _hash is in the cache
        :param _hash: hash to check
        :return: True if in Cache, False if not.
        """
        cache_hit = _hash in self._entries
        return cache_hit

    def _add_to_cache(self, _hash: str, filepath: str) -> None:
        # add entry to the cache.
        self._entries[_hash] = filepath

    def remove(self, uri: str) -> None:
        """
        Remove an entry from the cache
        :param uri: uri
        :return: None
        """
        if not self.in_cache(uri):
            raise ValueError(f'Key {uri} not in Cache')

        _hash = self._cache_file_name(uri)
        return self._remove_item_from_cache(_hash)

    def _remove_item_from_cache(self, _hash: str) -> None:
        """
        Remove a hash key from the cache. Here it is assumed that the _hash is
        a valid entry. We do allow for non existance of corresponding files as
        the cache can get out of sync if something external deleted the file.
        Since the endstate is valid (no entry in cache, no entry on disk) this
        is considered OK.

        :param _hash: hash key
        :return: None
        """

        assert _hash in self._entries

        file_to_delete = self._entries.pop(_hash)

        if os.path.exists(file_to_delete):
            logger.debug(f' - removing {_hash}')

            # And delete file.
            os.remove(file_to_delete)
        else:
            logger.debug(f' - file {_hash} did not exist on disk')

        return None

    def _get_from_cache(self, _hash: str) -> str:
        """
        Get entry from cache and touch the file to indicate it has been used
        recently (last to be evicted)
        :param _hash: file_hash corresponding to uri
        :return: file path
        """
        filepath = self._entries[_hash]

        if not os.path.exists(filepath):
            raise FileNotFoundError('The filepath in the cache log does not'
                                    'exist on disk.')

        # Touch the file to indicate we recently used it.
        Path(filepath).touch()

        return filepath

    def __len__(self) -> int:
        """
        :return: Number of entries in the cache.
        """
        return len(self._entries)

    def __getitem__(self, uris: Union[List, str]) -> List[str]:

        # make sure input is a list
        if isinstance(uris, str):
            uris = [uris]

        # Create the hashes from the URI's
        hashes = [self._cache_file_name(uri) for uri in uris]

        # collect key, full local path and hash for any hashes that are not
        # in the local cache
        cache_misses = []
        for _hash, key in zip(hashes, uris):
            to_append = (key, self._cache_file_path(key), _hash)
            if self._is_in_cache(_hash):
                if not self._validate( self._entries[_hash] ):
                    self._remove_item_from_cache(_hash)
                    cache_misses.append(to_append)
            else:
                cache_misses.append(to_append)

        self._cache_misses += len(cache_misses)
        self._cache_hits += len(hashes) - len(cache_misses)

        # for all URI's not in cache
        if cache_misses:
            # download all the files
            succesfully_downloaded = _download_from_resources(
                cache_misses,
                self.resources,
                parallel_download=self.parallel,
                allow_for_missing_files=self.allow_for_missing_files)

            # For all downloaded files do
            for success, cache_miss in zip(
                    succesfully_downloaded, cache_misses):
                if success:
                    # If succesfull, add to cache.
                    self._post_process(cache_miss[1])
                    self._add_to_cache(cache_miss[2], cache_miss[1])
                else:
                    # If not succesful, remove from keys to return
                    # Todo some logging or erroring?
                    hashes.pop(hashes.index(cache_miss[2]))

        # Get the filepaths to return
        filepaths = [self._get_from_cache(_hash) for _hash in hashes]

        size_of_requested_data = _get_total_size_of_files_in_bytes(filepaths)
        if size_of_requested_data > self.max_size:
            warning = f'The requested data does not fit into the cache.' \
                      f'To avoid issues the cache is enlarged to ensure' \
                      f'the current set of files fits in the cache. \n' \
                      f'old size: {self.max_size} bytes; ' \
                      f'new size {size_of_requested_data + MEGABYTE}'
            warn(warning)
            logger.warning(warning)
            self.max_size = size_of_requested_data + MEGABYTE

        # See if we need to do any cache eviction because the cache has become
        # to big.
        self._cache_eviction()
        return filepaths

    def _cache_eviction(self) -> bool:
        """
        Simple cache eviction policy. If the cache exceeds the maximum size
        remove data from the cache based on whichever file was interacted with
        the longest time ago. Evict files until we are below the acceptable
        cache size.

        :return: True if eviction occured, False otherwise.
        """

        # check if we exceed the size, if not return
        if not self._size() > self.max_size:
            return False

        # Get access/modified times for all the files in cache
        modified = []
        for _hash, fp in self._entries.items():
            # From my brief reading, access time is not always reliable,
            # hence I use whatever the latest time set is for modified or
            # access time as an indicator of when we last interacted with
            # the file.
            access_time = os.path.getatime(fp)
            modified_time = os.path.getmtime(fp)

            # pick whichever is most recent.
            time_to_check = access_time if access_time > modified_time \
                else modified_time
            modified.append((time_to_check, _hash))

        # Sort files in reversed chronological order.
        files_in_cache = [x[1] for x in
                          sorted(modified, key=lambda x: x[0], reverse=True)]

        # Delete files one by one as long as the cache_size exceeds the max
        # size.
        while (_size := self._size()) > self.max_size:
            self._cache_evictions += 1
            logger.debug(
                f'Cache exceeds limits: {_size} bytes, max size: '
                f'{self.max_size} bytes')

            # Get the hash and path of the oldest file and remove
            self._remove_item_from_cache(files_in_cache.pop())

        return True

    def _size(self) -> int:
        """
        Return size on disk of the cache in bytes.
        :return: cache size in bytes.
        """
        return _get_total_size_of_files_in_bytes(
            list(self._entries.values()), self.path)

    def purge(self) -> None:
        """
        Delete all the files in the cache.
        :return: None
        """
        logger.debug(f'Purging cache')
        keys = list(self._entries.keys())
        for key in keys:
            filepath = self._entries.pop(key)
            logger.debug(f' - deleting {filepath}')
            os.remove(filepath)
        logger.debug(f'Purging cache done')


def _download_from_resources(key_and_filenames: List[Tuple],
                             resources: List[RemoteResource],
                             parallel_download=False,
                             allow_for_missing_files=False) -> List[bool]:
    """
    Wrapper function to download multiple uris from the resource(s).

    :param key_and_filenames: List containing (uri,filename)
    :param parallel_download: If true, downloading is performed in parallel.
    :return: List of boolean indicating if the download was a success.
    """

    # construct the arguments to be used for parallel downloading of files.
    # Specifically, we need to match the right resource for downloading to the
    # right URI.
    args = []
    for key_and_filename in key_and_filenames:
        # Loop over all resources until we find one that can interpret the URI
        # (this is pretty naive approach and should probably be refactored to
        #  some direct mapping if the number of resources ever gets very long)
        for resource in resources:
            # For each resource check if the resource can interpret the URI
            if resource.valid_uri(key_and_filename[0]):
                # If so, get the download function, and other arguments and
                # break
                args.append(
                    (resource.download(), *key_and_filename[0:2],
                     allow_for_missing_files)
                )
                break
        else:
            # If we didn't break the loop no valid resource was found, raise
            # error
            raise ValueError(f'No resource available for URI: '
                             f'{key_and_filename[0]}')

    # Download the requested objects.
    if parallel_download:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            output = list(tqdm(pool.imap(_worker, args), total=len(args)))
    else:
        output = list(tqdm(map(_worker, args),total=len(args)))

    return output


def _get_total_size_of_files_in_bytes(filenames: List[str], path=None) -> int:
    """
    Simple function to calculate the size of a list of files on disk.
    :param filenames: list of filenames or filepaths
    :param path: if filenames are provided, this lists the path, otherwise set
        to None

    :return: Total size in bytes
    """
    size = 0
    for filename in filenames:
        if path is None:
            filepath = filename
        else:
            filepath = os.path.join(path, filename)
        size += os.path.getsize(filepath)
    return size


def _hashname(string: str) -> str:
    """
    Returns a md5 hash of a given string.
    :param string: input string
    :return: hexdigest of md5 hash.
    """
    return hashlib.md5(string.encode(), usedforsecurity=False).hexdigest()


def _worker(args) -> bool:
    download_func = args[0]
    uri = args[1]
    filepath = args[2]
    allow_for_missing_files = args[3]
    try:
        download_func(uri, filepath)
        return True
    except _RemoteResourceUriNotFound as e:
        if allow_for_missing_files:
            warning = f'Uri not retrieved: {str(e)}'
            warn(warning)
            logger.warning(warning)
        else:
            raise e
        return False