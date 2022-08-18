"""
Contents: Simple local caching of spectral requests.

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
from botocore.exceptions import ClientError
from roguewave.filecache.exceptions import _RemoteResourceUriNotFound
from boto3 import resource
from typing import Sequence,List, Iterable
# Constants
# =============================================================================


# Spotter Cache setup.
CACHE_NAME = 'spectral_cache'
CACHE_PATH = '~/temporary_roguewave_files/spectral_cache'
CACHE_SIZE_GB = 5


# Classes
# =============================================================================


class RestartFileResource(RemoteResource):
    """
    Remote resource for downloading data from restart file data given a "URI".
    """
    URI_PREFIX = 'restart-file-s3://'

    def __init__(self):
        self.s3 = resource('s3')

    def download(self):
        def _download_file_from_aws(uri: str, filepath: str) -> bool:
            """
            Worker function to download files from s3. Raise error if the
            object does not exist on s3.
            :param uri: valid uri for resource
            :param filepath: valid filepath to download remote object to.
            :return: True on success
            """
            s3 = self.s3
            decoded_uri = _decode_restart_file_cache_uri(uri, self.URI_PREFIX)
            obj = s3.Object(decoded_uri['bucket'], decoded_uri['key'])

            try:
                if decoded_uri["start"] == 0 and decoded_uri["stop"] == -1:
                    response = obj.download_file(filepath)
                else:
                    data = obj.get(Range=f'bytes={decoded_uri["start"]}-'
                                     f'{decoded_uri["stop"]-1}')['Body']
                    with open(filepath,'wb') as file:
                        file.write(data.read())

            except ClientError as e:
                raise _RemoteResourceUriNotFound(
                    f'Error downloading from {uri}. \n'
                    f'Error code: {e.response["Error"]["Code"]} '
                    f'Error message: {e.response["Error"]["Message"]}'
                )
            return True

        return _download_file_from_aws


# Functions
# =============================================================================
def get_data(uri:Sequence, start_byte:Sequence, stop_byte:Sequence,
                 cache_name: str = None
             ) -> List[bytearray]:
    """

    :return:
    """
    if cache_name is None:
        cache_name = CACHE_NAME

    if not _exists():
        _create_cache(cache_name)

    uris = [_restart_file_cache_uri(x, y, z) for
            x,y,z in zip(uri,start_byte,stop_byte)]

    filepaths = filecache.filepaths(uris, cache_name=cache_name)[0]
    if isinstance(filepaths,str):
        filepaths = [filepaths]

    output = []
    for filepath in filepaths:
        with open( filepath,'rb') as file:
            output.append(bytearray(file.read()))
    return output


# Module internal functions
# =============================================================================
def _restart_file_cache_uri(uri:str, start_byte:int, stop_byte:int) -> str:
    return f'restart-file-{uri}:{start_byte},{stop_byte}'

def _decode_restart_file_cache_uri(uri:str, uri_prefix) -> dict:
    bucket, key_byterange = uri.replace(uri_prefix, '').split('/', maxsplit=1)
    key, byterange = key_byterange.split(':')
    start, stop = byterange.split(',')
    return {'bucket':bucket, 'key':key, 'start':int(start),'stop':int(stop)}

def _create_cache(cache_name):
    if not _exists():
        restartfileresource = RestartFileResource()
        filecache.create_cache(cache_name,
                               cache_path=CACHE_PATH,
                               resources=[restartfileresource],
                               cache_size_GB=CACHE_SIZE_GB)


def _exists() -> bool:
    """
    Check if a spotter cache already exists
    :return: True or False
    """
    return filecache.exists(CACHE_NAME)