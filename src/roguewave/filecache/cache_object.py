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
from typing import Union, List, Tuple, Callable, Dict
from dataclasses import dataclass
from tqdm import tqdm
from roguewave import logger
from .remote_resources import (
    RemoteResourceS3,
    RemoteResourceHTTPS,
    RemoteResource,
    RemoteResourceLocal,
)
from .exceptions import _RemoteResourceUriNotFound
from json import dumps, load

TEMPORARY_DIRECTORY = "~/temporary_roguewave_files/filecache/"
CACHE_SIZE_GB = 5
MAXIMUM_NUMBER_OF_WORKERS = 10
KILOBYTE = 1000
MEGABYTE = 1000 * KILOBYTE
GIGABYTE = 1000 * MEGABYTE


@dataclass()
class CacheMiss:
    uri: str
    filepath: str
    filename: str
    allow_for_missing_files: bool
    post_process_function: Callable[[str], None]
    download_function: Callable[[str, str], bool] = None


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

    CACHE_FILE_PREFIX = "cachefile_"
    CACHE_FILE_POSTFIX = "_cachefile"

    def __init__(
        self,
        path: str = TEMPORARY_DIRECTORY,
        size_GB: Union[float, int] = CACHE_SIZE_GB,
        do_cache_eviction_on_startup: bool = False,
        resources: List[RemoteResource] = None,
        parallel=True,
        allow_for_missing_files=False,
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
        # create the path if it does not exist
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.config = {
            "size_gb": size_GB,
            "parallel": parallel,
            "allow_for_missing_files": allow_for_missing_files,
        }
        if self.config_exists():
            self.config |= self.load_config()
        else:
            self._write_config()

        # Some counters to keep track of total cache misses, hits and
        # evictions. No downstream use right now/
        self._cache_misses = 0
        self._cache_hits = 0
        self._cache_evictions = 0
        self.disable_progress_bar = False

        # initialize the cache.
        self._entries = {}  # the key/value pair cache
        self._initialize_cache(do_cache_eviction_on_startup)

        # Post processing and validation functions
        self.directives = {"validate": {}, "postprocess": {}}

        # message to display on progress bar
        self.description = "Caching"

        # download resources
        if resources is None:
            self.resources = [
                RemoteResourceS3(),
                RemoteResourceHTTPS(),
                RemoteResourceLocal(),
            ]
        else:
            self.resources = resources

    @property
    def config_name(self) -> str:
        return os.path.join(self.path, "file_cache_config.json")

    def config_exists(self) -> bool:
        return os.path.exists(self.config_name)

    def load_config(self) -> Dict:
        with open(self.config_name, "rb") as fp:
            return load(fp)

    def _update_config(self, key, value, write=True):
        self.config[key] = value
        if write:
            self._write_config()

    def _write_config(self):
        with open(os.path.join(self.path, "file_cache_config.json"), "wt") as fp:
            fp.write(dumps(self.config, indent=4))

    @property
    def max_size(self) -> int:
        return self.config["size_gb"]

    @max_size.setter
    def max_size(self, size_gb: float):
        self._update_config("size_gb", size_gb)

    @property
    def max_size_bytes(self) -> int:
        return int(self.config["size_gb"] * GIGABYTE)

    @max_size_bytes.setter
    def max_size_bytes(self, size_bytes: int):
        self._update_config("size_gb", size_bytes / GIGABYTE)

    @property
    def parallel(self) -> bool:
        return self.config["parallel"]

    @parallel.setter
    def parallel(self, parallel: bool):
        self._update_config("parallel", parallel)

    @property
    def allow_for_missing_files(self) -> bool:
        return self.config["allow_for_missing_files"]

    @allow_for_missing_files.setter
    def allow_for_missing_files(self, allow_for_missing_files: bool):
        self._update_config("allow_for_missing_files", allow_for_missing_files)

    def set_directive_function(
        self,
        directive,
        name,
        function: Union[Callable[[str], None], Callable[[str], bool]],
    ):

        if directive not in self.directives:
            raise KeyError(f"{directive} is not a valid cache directive.")

        if name in self.directives[directive]:
            raise ValueError(f"Function  for {name} already exists")
        else:
            self.directives[directive][name] = function

    def remove_directive_function(self, directive: str, name: str):

        if directive not in self.directives:
            raise KeyError(f"{directive} is not a valid cache directive.")

        if name not in self.directives[directive]:
            raise ValueError(f"Function  for {name} does not exist")
        else:
            self.directives[directive].pop(name)

    def _cache_file_name(self, uri: str) -> str:
        """
        Return the filename that corresponds to the given uri. We construct
        the file name using a simple md5 hash of the uri string prefixed
        with a cache file prefix. THe later is introduced to seperate cache
        files in a path from user files (and avoid including/deleting those).

        :param uri: valid uri stripped from directives
        :return: valid cache file
        """
        return self.CACHE_FILE_PREFIX + _hashname(uri) + self.CACHE_FILE_POSTFIX

    def _cache_file_path(self, uri: str) -> str:
        """
        Construct the path where the given uri is stored locally.

        :param uri: valid uri stripped from directives.
        :return: valid cache file
        """
        return os.path.join(self.path, self._cache_file_name(uri))

    def _get_cache_files(self) -> List[str]:
        """
        Find all files that are currently a member of the cache.
        :return:
        """
        for path, dirs, files in os.walk(self.path):
            # Return all files that are "cache" objects. This is a safety if
            # other user files are present, so that these don't accidentally
            # evicted from the cache (aka deleted).
            return [
                file
                for file in files
                if file.startswith(self.CACHE_FILE_PREFIX)
                and file.endswith(self.CACHE_FILE_POSTFIX)
            ]
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
            if self._size() > self.max_size_bytes:
                raise ValueError(
                    "The cache currently existing on disk "
                    "exceeds the maximum cache size of the "
                    f"current cache ({self.max_size} gb)."
                    f"\n The cache size can be increased by"
                    f" editting the cache config file: "
                    f"{self.config_name}"
                )

    def in_cache(self, unparsed_uris) -> List[bool]:
        # make sure input is a list
        if isinstance(unparsed_uris, str):
            unparsed_uris = [unparsed_uris]

        uris, _ = parse_directives(unparsed_uris)

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

    def remove(self, unparsed_uri: str) -> None:
        """
        Remove an entry from the cache
        :param unparsed_uri: uri
        :return: None
        """
        uri, _ = parse_directive(unparsed_uri)

        if not self.in_cache(uri):
            raise ValueError(f"Key {uri} not in Cache")

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

        if _hash not in self._entries:
            return None

        file_to_delete = self._entries.pop(_hash)

        if os.path.exists(file_to_delete):
            logger.debug(f" - removing {_hash}")

            # And delete file.
            os.remove(file_to_delete)
        else:
            logger.debug(f" - file {_hash} did not exist on disk")

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
            raise FileNotFoundError(
                "The filepath in the cache log does not" "exist on disk."
            )

        # Touch the file to indicate we recently used it.
        Path(filepath).touch()

        return filepath

    def get_cache_misses(
        self, uris: List[str], directives: List[Dict[str, str]]
    ) -> List[CacheMiss]:
        """
        Function to get all cache misses and return a list of CacheMiss objects
        needed to download the misses from remote resources.

        This function also perform validates on potential cache hits if a
        relevant validation function is set *and* validation is requested
        through a directive.

        :param uris: list of uris stripped of directives
        :param directives: list of directives per uri (empty dict if none)
        :return: list of cache misses
        """

        cache_misses = []
        for uri, directive in zip(uris, directives):

            # what is the hashkey/filename
            hashkey = self._cache_file_name(uri)
            filepath = self._cache_file_path(uri)

            # is the key in cache?
            valid_entry = False
            if self._is_in_cache(hashkey):
                # If so is it a valid entry
                if "validate" in directive:
                    # Call the user supplied validation function with the
                    # filepath as argument
                    validation_function = self.directives["validate"][
                        directive["validate"]
                    ]

                    try:
                        valid_entry = validation_function(filepath)
                    except IOError:
                        valid_entry = False

                    if not valid_entry:
                        # remove the locally stored entry if not valid
                        os.remove(filepath)
                    else:
                        valid_entry = True
                else:
                    # Defaults to True if no validation directive is given
                    valid_entry = True

            if not valid_entry:
                # If not a valid entry (either missing or invalid)
                #
                if "postprocess" in directive:
                    # Add the postprocess function to use if requested.
                    post_process_function = self.directives["postprocess"][
                        directive["postprocess"]
                    ]

                else:
                    # otherwise set a null function as postprocessor
                    post_process_function = do_nothing

                cache_misses.append(
                    CacheMiss(
                        uri=uri,
                        filepath=filepath,
                        filename=hashkey,
                        allow_for_missing_files=self.allow_for_missing_files,
                        post_process_function=post_process_function,
                    )
                )
        return cache_misses

    def __len__(self) -> int:
        """
        :return: Number of entries in the cache.
        """
        return len(self._entries)

    def __getitem__(self, unparsed_uris: Union[List, str]) -> List[str]:
        """
        Get filenames corresponding to locally stored versions of the objects
        the URI points to. Note that the unparsed_uris take the form:

        [ directive=option ; ... directive=option ] ":" [scheme] "://" [path]

        e.g for amazon s3 where we want to perform validation and post
            processing on entries:

            validate=grib;postprocess=grib:s3://bucket/key

        or without cache directives

            s3://bucket/key

        Cache directives are optional, but if specified the corresponding
        user defined handling function must have been set. By default no
        validation or postprocessing functions are set.

        :param unparsed_uris: URI's that may still include directives.
        :return:
        """
        # make sure input is a list
        if isinstance(unparsed_uris, str):
            unparsed_uris = [unparsed_uris]

        # Remove cache directives from uris (if included)
        uris, directives = parse_directives(unparsed_uris)
        filepaths = [self._cache_file_path(uri) for uri in uris]

        # for all URI's not in cache
        if cache_misses := self.get_cache_misses(uris, directives):
            _ = _download_from_resources(
                cache_misses,
                self.resources,
                parallel_download=self.parallel,
                disable_progress_bar=self.disable_progress_bar,
                desc=self.description,
            )

            for cache_miss in cache_misses:
                self._add_to_cache(cache_miss.filename, cache_miss.filepath)

        size_of_requested_data = _get_total_size_of_files_in_bytes(filepaths)
        if size_of_requested_data > self.max_size_bytes:
            warning = (
                f"The requested data does not fit into the cache."
                f"To avoid issues the cache is enlarged to ensure"
                f"the current set of files fits in the cache. \n"
                f"old size: {self.max_size_bytes} bytes; "
                f"new size {size_of_requested_data + MEGABYTE}"
            )
            warn(warning)
            logger.warning(warning)
            self.max_size_bytes = size_of_requested_data + MEGABYTE

        self._cache_misses += len(cache_misses)
        self._cache_hits += len(uris) - len(cache_misses)

        # See if we need to do any cache eviction because the cache has become
        # to big.
        if not len(cache_misses) == 0:
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
        if not self._size() > self.max_size_bytes:
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
            time_to_check = (
                access_time if access_time > modified_time else modified_time
            )
            modified.append((time_to_check, _hash))

        # Sort files in reversed chronological order.
        files_in_cache = [
            x[1] for x in sorted(modified, key=lambda x: x[0], reverse=True)
        ]

        # Delete files one by one as long as the cache_size exceeds the max
        # size.
        while (_size := self._size()) > self.max_size_bytes:
            self._cache_evictions += 1
            logger.debug(
                f"Cache exceeds limits: {_size} bytes, max size: "
                f"{self.max_size_bytes} bytes"
            )

            # Get the hash and path of the oldest file and remove
            self._remove_item_from_cache(files_in_cache.pop())

        return True

    def _size(self) -> int:
        """
        Return size on disk of the cache in bytes.
        :return: cache size in bytes.
        """
        return _get_total_size_of_files_in_bytes(
            list(self._entries.values()), self.path
        )

    def purge(self) -> None:
        """
        Delete all the files in the cache.
        :return: None
        """
        logger.debug("Purging cache")
        keys = list(self._entries.keys())
        for key in keys:
            filepath = self._entries.pop(key)
            logger.debug(" - deleting {filepath}")
            os.remove(filepath)
        logger.debug("Purging cache done")


def _download_from_resources(
    cache_misses: List[CacheMiss],
    resources: List[RemoteResource],
    parallel_download=False,
    disable_progress_bar=False,
    desc="",
) -> List[bool]:
    """
    Wrapper function to download multiple uris from the resource(s).
    :param cache_misses: List containing cache misses to download
    :param parallel_download: If true, downloading is performed in parallel.
    :return: List of boolean indicating if the download was a success.
    """

    def _worker(cache_miss: CacheMiss) -> bool:
        try:
            cache_miss.download_function(cache_miss.uri, cache_miss.filepath)
            cache_miss.post_process_function(cache_miss.filepath)
            return True
        except _RemoteResourceUriNotFound as e:
            if cache_miss.allow_for_missing_files:
                warning = f"Uri not retrieved: {str(e)}"
                warn(warning)
                logger.warning(warning)
            else:
                raise e
            return False

    # construct the arguments to be used for parallel downloading of files.
    # Specifically, we need to match the right resource for downloading to the
    # right URI.
    for cache_miss in cache_misses:
        # Loop over all resources until we find one that can interpret the URI
        # (this is pretty naive approach and should probably be refactored to
        #  some direct mapping if the number of resources ever gets very long)
        for resource in resources:
            # For each resource check if the resource can interpret the URI
            if resource.valid_uri(cache_miss.uri):
                # If so, get the download function, and other arguments and
                # break
                cache_miss.download_function = resource.download()
                break
        else:
            # If we didn't break the loop no valid resource was found, raise
            # error
            raise ValueError(f"No resource available for URI: " f"{cache_miss.uri}")

    # Download the requested objects.
    if parallel_download and len(cache_misses) > 0:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            output = list(
                tqdm(
                    pool.imap(_worker, cache_misses), desc=desc, total=len(cache_misses)
                )
            )
    else:
        if len(cache_misses) == 1:
            disable_progress_bar = True

        output = list(
            tqdm(
                map(_worker, cache_misses),
                total=len(cache_misses),
                disable=disable_progress_bar,
                desc=desc,
            )
        )
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


def parse_directives(raw_uris: List[str]) -> Tuple[List[str], List[dict]]:
    uris = []
    directives = []
    for raw_uri in raw_uris:
        uri, directive = parse_directive(raw_uri)
        uris.append(uri)
        directives.append(directive)
    return uris, directives


def parse_directive(unparsed_uri: str) -> Tuple[str, dict]:
    """
    unparsed_uris take the form:

        [ directive=option ; ... directive=option ] ":" [scheme] "://" [path]

        e.g for amazon s3 where we want to perform validation and post
            processing on entries:

            validate=grib;postprocess=grib:s3://bucket/key

        or without cache directives

            s3://bucket/key

    This function seperates the directive/option pairs into a directove
    dictionary, and a valid uri, i.e.

                validate=grib;postprocess=grib:s3://bucket/key

    becomes

        directive = { "validate":"grib", "postprocess":"grib}
        uri = s3://bucket/key

    The parsing is really simple.

    :param unparsed_uri: uri possibly containing cache directives
    :return:
    """

    # split in directives_scheme part and a path.
    directives_and_scheme, path = unparsed_uri.split("://")

    parsed_directives = {}
    # if a colon is present then directives are provided.
    if ":" in directives_and_scheme:
        # split directives from the scheme
        directives, scheme = directives_and_scheme.split(":")

        # split multiple directives (if present)
        directives = directives.split(";")

        # for each directive store in the dict.
        for directive in directives:
            directive_name, directive_parameter = directive.split("=")
            parsed_directives[directive_name] = directive_parameter
    else:
        # no directives
        scheme = directives_and_scheme

    uri = scheme + "://" + path
    return uri, parsed_directives


def do_nothing(*arg, **kwargs):
    """
    Null function for convenience
    :param arg:
    :param kwargs:
    :return:
    """
    return None
