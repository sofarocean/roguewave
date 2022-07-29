# Import
# =============================================================================
import os
import hashlib
import boto3
from typing import List, Tuple, Dict, Union
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from roguewave import logger
from pathlib import Path
from warnings import warn

# Model private variables
# =============================================================================


TEMPORARY_DIRECTORY = '~/temporary_roguewave_files/filecache/'
CACHE_SIZE_GiB = 1
MAXIMUM_NUMBER_OF_WORKERS = 10
KILOBYTE = 1024
MEGABYTE = 1024 * KILOBYTE
GIGABYTE = 1024 * MEGABYTE

_ACTIVE_FILE_CACHES = {}


# Classes
# =============================================================================


class AWSFileCache():
    """
    Simple file caching class that when given an aws key locally stores the
    file in the cache directory and returns the path to the file. The file
    remains in storage until the cache directory exceeds a prescribed size,
    in which case files with oldest access/modified dates get deleted first
    until everything fits in the cache again. Any time a file is accessed it's
    modified date gets updated so that often used files automaticall remain in
    cache.

    The files are stored locally in the directory specified on class
    initialization, as:

        [path]/CACHE_PREFIX + md5_hash_of_AWS_KEY

    Methods
      * __getitem__(keys) : accept a simgle aws_key or multiple keys and returns
        filepaths to local copies thereof. You would typically use the
            cache[keys] notation instead of the dunder method.
      * purge() clear all contents of the cache (destructive, deletes all local
        files).

    Usage:

        cache = FileCache()
        list_of_local_file_names = cache[ [list of aws_keys ] ]

    # do stuff with file
    ...
    """

    CACHE_FILE_PREFIX = 'cachefile_'

    def __init__(self,
                 path:str=TEMPORARY_DIRECTORY,
                 size_GiB:Union[float,int]=CACHE_SIZE_GiB,
                 do_cache_eviction_on_startup:bool=False,
                 parallel=True):
        """
        Initialize Cache
        :param path: path to store cache. If path does not exist it will be
            created.
        :param size_GiB: Maximum size of the cache in GiB. If cache exceeds
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
        self.max_size = int( size_GiB * GIGABYTE )
        self.parallel = parallel

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

    def _cache_file_name(self, aws_key: str) -> str:
        """
        Return the filename that corresponds to the given aws_key. We construct
        the file name using a simple md5 hash of the aws_key string prefixed
        with a cache file prefix. THe later is introduced to seperate cache
        files in a path from user files (and avoid including/deleting those).

        :param aws_key: valis aws_key of form "bucket/key"
        :return: valid cache file
        """
        return self.CACHE_FILE_PREFIX + _hashname(aws_key)

    def _cache_file_path(self, aws_key: str) -> str:
        """
        Construct the path where the given aws_key is stored locally.

        :param aws_key: valis aws_key of form "bucket/key"
        :return: valid cache file
        """
        return os.path.join(self.path,
                            self.CACHE_FILE_PREFIX + _hashname(aws_key))

    def _get_cache_files(self) -> List[str]:
        """
        Find all files that are currently a member of the cache
        :return:
        """
        _cache_files = []
        for path, dirs, files in os.walk(self.path):
            # Return all files that are "cache" objects. This is a safety if
            # other user files are present, so that these don't accidentally
            # evicted from the cache (aka deleted).
            return [file for file in files if self.CACHE_FILE_PREFIX if file]
        else:
            return []

    def _initialize_cache(self,do_cache_eviction_on_stratup:bool) -> None:
        """
        Initialize the file cache. Look on disk for files in the cache path
        that have the required prefix and load these into the cache. Once
        loaded, we do a check whether or not the cache is full and if we need
        to remove files.
        :param do_cache_eviction_on_stratup: see description under __init__
        :return:
        """
        self._entries = {}
        for file in self._get_cache_files():
            filepath = os.path.join(self.path, file)
            self._entries[file] = filepath

        # See if cache is still < size
        if do_cache_eviction_on_stratup:
            self._cache_eviction()
        else:
            if self._size() > self.max_size:
                raise ValueError( 'The cache currently existing on disk '
                                  'exceeds the maximum cache size of the '
                                  'current cache.\n Either increase the cache '
                                  'size of the object or allow eviction of '
                                  'files on startup.' )

    def _is_in_cache(self, _hash: str) -> bool:
        """
        Check if a _hash is in the cache
        :param _hash:
        :return:
        """
        cache_hit = _hash in self._entries
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return cache_hit

    def _add_to_cache(self, _hash: str, filepath: str) -> None:
        # add entry to the cache
        self._entries[_hash] = filepath

    def _get_from_cache(self, _hash:str ) -> str:
        """
        Get entry from cache and touch the file to indicate it has been used
        recently (last to be evicted)
        :param _hash: file_hash corresponding to aws key
        :return: file path
        """
        filepath = self._entries[_hash]
        # Touch the file to indicate we recently used it.
        Path(filepath).touch()
        return filepath

    def __len__(self) -> int:
        """
        :return: Number of entries in the cache.
        """
        return len(self._entries)


    def __getitem__(self, aws_keys: Union[List, str]) -> List[str]:

        # make sure input is a list
        if isinstance(aws_keys, str):
            aws_keys = [aws_keys]

        # Create the hashes from the aws keys
        hashes = [self._cache_file_name(aws_key) for aws_key in aws_keys]

        # collect key, full local path and hash for any hashes that are not
        # in the local cache
        cache_misses = [(key, self._cache_file_path(key), _hash) for
                        _hash, key in zip(hashes, aws_keys) if
                        not self._is_in_cache(_hash)]

        # for all aws_keys not in cache
        if cache_misses:
            # download all the files
            succesfully_downloaded = _download_from_aws(cache_misses,
                                            parallel_download=self.parallel)

            # For all downloaded files do
            for success, cache_miss in zip(
                    succesfully_downloaded, cache_misses):
                if success:
                    # If succesfull, add to cache.
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
                      f'new size {size_of_requested_data+MEGABYTE}'
            warn(warning)
            logger.warning(warning)
            self.max_size = size_of_requested_data + MEGABYTE

        # See if we need to do any cache eviction because the cache has become
        # to big.
        self._cache_eviction()
        return filepaths

    def _cache_eviction(self)->bool:
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
        for path, dirs, files in os.walk(self.path):
            for _hash in files:
                fp = os.path.join(path, _hash)

                # From my brief reading, access time is not always reliable,
                # hence I use whatever the latest time set is for modified or
                # access time as an indicator of when we last interacted with
                # the file.
                access_time = os.path.getatime(fp)
                modified_time = os.path.getmtime(fp)

                time_to_check = access_time if access_time > modified_time else modified_time
                modified.append((time_to_check, fp, _hash))

        # Sort files in reversed chronological order.
        files_in_cache = [(x[1], x[2]) for x in
                          sorted(modified, key=lambda x: x[0], reverse=True)]

        # Delete files one by one as long as the cache_size exceeds the max
        # size.
        while (_size := self._size()) > self.max_size:
            self._cache_evictions += 1
            logger.debug(
                f'Cache exceeds limits: {_size} bytes, max size: '
                f'{self.max_size} bytes')

            # Get the hash and path of the oldest file
            file_to_delete, _hash = files_in_cache.pop()

            # remove from cache entries.
            self._entries.pop(_hash)
            logger.debug(f' - removing {_hash}')

            # And delete file.
            os.remove(file_to_delete)

        return True

    def _size(self)->int:
        """
        Return size on disk of the cache in bytes.
        :return: cache size in bytes.
        """
        size = 0
        for path, dirs, files in os.walk(self.path):
            size = _get_total_size_of_files_in_bytes(files,path)
        return size

    def purge(self)->None:
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


# Main public function.
# =============================================================================


def cached_local_aws_files( aws_keys:List[str], cache_name:str)-> List[str]:
    """

    :param aws_keys:
    :param cache_name:
    :return:
    """

    if cache_name not in _ACTIVE_FILE_CACHES:
        raise ValueError(f'Cache with name {cache_name} does not exist.')

    return _ACTIVE_FILE_CACHES[cache_name][aws_keys]


def create_aws_file_cache(cache_name:str,
                      cache_path:str=TEMPORARY_DIRECTORY,
                      cache_size_GiB:Union[int,float]=CACHE_SIZE_GiB,
                      do_cache_eviction_on_startup:bool=False,
                      download_in_parallel=True
                      )  \
        -> None:
    """
    Create a file cache. Created caches *must* have unique names and
    cache_paths.

    :param cache_name: name for the cache to be created. This name is used
            to retrieve files from the cache.
    :param cache_path: path to store cache. If path does not exist it will be
            created.
    :param cache_size_GiB:  Maximum size of the cache in GiB. If cache exceeds
            the size, then files with oldest access/modified dates get deleted
            until everthing fits in the cache again. Fractional values (floats)
            are allowed.
    :param do_cache_eviction_on_startup: do_cache_eviction_on_startup: whether we ensure the cache size
            conforms to the given size on startup. If set to true, a cache
            directory that exceeds the maximum size will be reduced to max
            size. Set to False by default in which case an error occurs. The
            latter to prevent eroneously evicting files from a cache that was
            previously created on purpose with a larger size that the limit.
    :param download_in_parallel: Download in paralel from aws. Per default 10
            worker threads are created.

    :return:
    """
    cache_path = os.path.abspath(os.path.expanduser(cache_path))

    if cache_name in _ACTIVE_FILE_CACHES:
        raise ValueError(f'Cache with name {cache_name} is already initialized')

    for key,cache in _ACTIVE_FILE_CACHES.items():
        if cache.path == cache_path:
            raise ValueError(f'Error when creating cache with name: '
                             f'"{cache_name}". \n A cache named: "{key}" '
                             f'already uses the path {cache_path} '
                             f'for caching.\n '
                             f'Multiple caches cannot share the same path.')


    _ACTIVE_FILE_CACHES[cache_name] = AWSFileCache(
            cache_path,
            size_GiB=cache_size_GiB,
            do_cache_eviction_on_startup=do_cache_eviction_on_startup,
            parallel=download_in_parallel
    )
    return


def delete_aws_file_cache(cache_name):
    """
    Delete all files associated with a cache and remove cache from available
    caches. To note: all files are deleted, but the folder itself is not.

    :param cache_name: Name of the cache to be deleted
    :return:
    """
    if cache_name not in _ACTIVE_FILE_CACHES:
        raise ValueError(f'Cache with name {cache_name} does not exist')

    cache = _ACTIVE_FILE_CACHES.pop(cache_name)
    cache.purge()


# Private module helper functions
# =============================================================================


def _hashname(string: str) -> str:
    """
    Returns a md5 hash of a given string.
    :param string: input string
    :return: hexdigest of md5 hash.
    """
    return hashlib.md5(string.encode(), usedforsecurity=False).hexdigest()


def _download_from_aws(key_and_filenames: List[Tuple],
                       parallel_download=False)->List[bool]:
    """
    Wrapper function to download multiple aws_keys from aws.

    :param key_and_filenames: List containing (aws_key,filename)
    :param parallel_download: If true, downloading is performed in parallel.
    :return: List of boolean indicating if the download was a success.
    """
    s3 = boto3.client('s3')
    args = [ (s3,*key_and_filename) for key_and_filename in key_and_filenames ]
    if parallel_download:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            output = list(
                tqdm(pool.imap(_download_file_from_aws, args),
                     total=len(args)))
    else:
        output = list(
            tqdm(map(_download_file_from_aws, args),
                 total=len(args)))
    return output


def _download_file_from_aws(args):
    """
    Worker function to download files from AWS
    :param args: Tuple/List with as first entry the aws key to download and
        as second entry the filepath to download to
    :return:
    """
    s3 = args[0]
    aws_key = args[1]
    filepath = args[2]
    bucket, key = aws_key.split('/', maxsplit=1)

    s3.download_file(bucket, key, filepath)
    return True

def _get_total_size_of_files_in_bytes(filenames:List[str], path = None) -> int:
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


def _get_cache( path, **kwargs )->AWSFileCache:
    if path in _ACTIVE_FILE_CACHES:
        return _ACTIVE_FILE_CACHES[path]
    else:
        _ACTIVE_FILE_CACHES[path] = AWSFileCache(
            path, **kwargs
        )