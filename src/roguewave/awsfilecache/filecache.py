# Import
# =============================================================================
import os
import hashlib
import boto3
from typing import List, Tuple, Union, Dict, Iterable, Callable
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from roguewave import logger
from pathlib import Path
from warnings import warn
from requests import get
from requests.exceptions import HTTPError
from botocore.exceptions import ClientError

# Model private variables
# =============================================================================


TEMPORARY_DIRECTORY = '~/temporary_roguewave_files/filecache/'

CACHE_SIZE_GiB = 5
DEFAULT_CACHE_NAME = '__default__'
MAXIMUM_NUMBER_OF_WORKERS = 10
KILOBYTE = 1024
MEGABYTE = 1024 * KILOBYTE
GIGABYTE = 1024 * MEGABYTE

_ACTIVE_FILE_CACHES = {}  # type: Dict[str,FileCache]


# Classes
# =============================================================================
class _RemoteResourceUriNotFound(Exception):
    pass


class RemoteResource():
    """
    Abstract class defining the resource protocol used for remote retrieval. It
    contains just two methods that need to be implemented:
    - download return a function that can download from the resource given a
      uri and filepath
    - method to check if the uri is a valid uri for the given resource.
    """
    URI_PREFIX = 'uri://'

    def download(self)-> Callable[[str, str], bool]:
        """
        Return a function that takes uri (first argument) and filepath (second
        argument), and downloads the given uri to the given filepath. Return
        True on Success. Raise _RemoteResourceUriNotFound if URI does not
        exist on the resource.
        """
        pass

    def valid_uri(self, uri: str) -> bool:
        """
        Check if the uri is valid for the given resource
        :param uri: Uniform Resource Identifier.
        :return: True or False
        """
        if uri.startswith(self.URI_PREFIX):
            return True
        else:
            return False

    def _remove_uri_prefix(self, uri: str):
        return uri.strip(self.URI_PREFIX)


class RemoteResourceS3(RemoteResource):
    URI_PREFIX = 's3://'

    def __init__(self):
        self.s3 = boto3.client('s3')

    def download(self):
        def _download_file_from_aws(uri: str, filepath: str)->bool:
            """
            Worker function to download files from s3. Raise error if the
            object does not exist on s3.
            :param uri: valid uri for resource
            :param filepath: valid filepath to download remote object to.
            :return: True on success
            """
            s3 = self.s3
            bucket, key = uri.replace(self.URI_PREFIX,'').split('/', maxsplit=1)
            try:
                s3.download_file(bucket, key, filepath)
            except ClientError as e:
                raise _RemoteResourceUriNotFound(
                    f'Error downloading from {uri}. \n'
                    f'Error code: {e.response["Error"]["Code"]} '
                    f'Error message: {e.response["Error"]["Message"]}'
                )
            return True

        return _download_file_from_aws


class RemoteResourceHTTPS(RemoteResource):
    URI_PREFIX = 'https://'

    def download(self):
        def _download_file_from_https(uri: str, filepath: str)->bool:
            """
            Worker function to download files from https url. Raise error if
            the object does not exist on s3.
            :param uri: valid uri for resource
            :param filepath: valid filepath to download remote object to.
            :return: True on success
            """
            try:
                response = get(uri, allow_redirects=True)
                status_code = response.status_code
                response.raise_for_status()
            except HTTPError as error:
                raise _RemoteResourceUriNotFound(
                    f"Error downloading from: {uri}, "
                    f"http status code: {status_code},"
                    f" message: {response.text}"
                )

            with open(filepath, 'wb') as file:
                file.write(response.content)

            return True

        return _download_file_from_https


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
                 size_GiB: Union[float, int] = CACHE_SIZE_GiB,
                 do_cache_eviction_on_startup: bool = False,
                 resources=None,
                 parallel=True,
                 allow_for_missing_files=False):
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
        self.max_size = int(size_GiB * GIGABYTE)
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

        # download resources
        if resources is None:
            self.resources = [RemoteResourceS3(), RemoteResourceHTTPS()]
        else:
            self.resources = resources

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
        cache_misses = [(key, self._cache_file_path(key), _hash) for
                        _hash, key in zip(hashes, uris) if
                        not self._is_in_cache(_hash)]

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


# Main public function.
# =============================================================================


def cached_local_files(
        uris: List[str],
        cache_name: str = None,
        return_cache_hits=False,
) -> Union[List[str], Tuple[List[str], List[bool]]]:
    """

    :param uris:
    :param cache_name:
    :return:
    """

    if cache_name is None:
        cache_name = DEFAULT_CACHE_NAME

    if cache_name not in _ACTIVE_FILE_CACHES:
        if cache_name == DEFAULT_CACHE_NAME:
            create_file_cache(cache_name)

        else:
            raise ValueError(f'Cache with name {cache_name} does not exist.')

    if return_cache_hits:
        cache_hits = _ACTIVE_FILE_CACHES[cache_name].in_cache(uris)
        return _ACTIVE_FILE_CACHES[cache_name][uris], cache_hits

    else:
        return _ACTIVE_FILE_CACHES[cache_name][uris]


def cache_exists(cache_name: str):
    """
    Check if the cache name already exists
    :param cache_name: name for the cache to be created. This name is used
            to retrieve files from the cache.
    :return: True if exists, False otherwise
    """
    return cache_name in _ACTIVE_FILE_CACHES


def get_cache(cache_name: str) -> FileCache:
    """
    Get a valid cache object, error if the name does not exist.
    :param cache_name: Name of the cache
    :return: Cache object
    """

    if cache_exists(cache_name):
        return _ACTIVE_FILE_CACHES[cache_name]
    else:
        raise KeyError(f'Cache with {cache_name} does not exist')


def create_file_cache(cache_name: str,
                      cache_path: str = TEMPORARY_DIRECTORY,
                      cache_size_GiB: Union[int, float] = CACHE_SIZE_GiB,
                      do_cache_eviction_on_startup: bool = False,
                      download_in_parallel=True
                      ) \
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
        raise ValueError(
            f'Cache with name {cache_name} is already initialized')

    for key, cache in _ACTIVE_FILE_CACHES.items():
        if cache.path == cache_path:
            raise ValueError(f'Error when creating cache with name: '
                             f'"{cache_name}". \n A cache named: "{key}" '
                             f'already uses the path {cache_path} '
                             f'for caching.\n '
                             f'Multiple caches cannot share the same path.')

    _ACTIVE_FILE_CACHES[cache_name] = FileCache(
        cache_path,
        size_GiB=cache_size_GiB,
        do_cache_eviction_on_startup=do_cache_eviction_on_startup,
        parallel=download_in_parallel
    )
    return


def delete_file_cache(cache_name):
    """
    Delete all files associated with a cache and remove cache from available
    caches. To note: all files are deleted, but the folder itself is not.

    :param cache_name: Name of the cache to be deleted
    :return:
    """
    if not cache_exists(cache_name):
        raise ValueError(f'Cache with name {cache_name} does not exist')

    cache = _ACTIVE_FILE_CACHES.pop(cache_name)
    cache.purge()


def delete_default_cache():
    """
    Clean up the default cache.

    :return:
    """
    if cache_exists(DEFAULT_CACHE_NAME):
        delete_file_cache(DEFAULT_CACHE_NAME)


def remove_cached_keys(uris: Union[str, Iterable[str]],
                       cache_name: str) -> None:
    """
    Remove given key(s) from the cache
    :param uris: list of keys to remove
    :param cache_name: name of initialized cache.
    :return:
    """
    if not isinstance(uris, Iterable):
        uris = [uris]

    cache = get_cache(cache_name)
    for key in uris:
        cache.remove(key)


# Private module helper functions
# =============================================================================


def _hashname(string: str) -> str:
    """
    Returns a md5 hash of a given string.
    :param string: input string
    :return: hexdigest of md5 hash.
    """
    return hashlib.md5(string.encode(), usedforsecurity=False).hexdigest()


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
    args = []
    for key_and_filename in key_and_filenames:
        for resource in resources:
            if resource.valid_uri(key_and_filename[0]):
                args.append(
                    (resource.download(), *key_and_filename[0:2],
                     allow_for_missing_files)
                )
                break
        else:
            raise ValueError(f'No resource available for URI: '
                             f'{key_and_filename[0]}')

    if parallel_download:
        with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
            output = list(
                tqdm(pool.imap(_worker, args),
                     total=len(args)))
    else:
        output = list(
            tqdm(map(_worker, args),
                 total=len(args)))
    return output


def _worker(args)->bool:
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
            warn( warning )
            logger.warning(warning)
        else:
            raise e
        return False


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
