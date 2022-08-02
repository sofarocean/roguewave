from roguewave.filecache.remote_resources import RemoteResource
from roguewave import filecache
from roguewave import save, load
from roguewave.io.io import NumpyEncoder, object_hook
import json
from typing import List

# Spotter Cache
CACHE_NAME = 'spotter_cache'
CACHE_PATH = '~/temporary_roguewave_files/spotter_cache'
CACHE_SIZE_GB = 2

class WaveFleetResource(RemoteResource):
    URI_PREFIX = 'wavefleet://'

    def __init__(self, request_type_handle_mapping:dict, session):
        self._handlers = request_type_handle_mapping
        self.session = session

    def download(self):
        def get_data_from_wavefleet( uri:str, filepath:str ):
            request_type, kwargs = decode_spotter_cache_uri(uri)
            data = self._handlers[request_type](session=self.session,**kwargs)
            save(data,filepath)
            return True
        return get_data_from_wavefleet

def spotter_cache_uri(request_type: str,
                      **kwargs):
    kwargs['request_type'] = request_type
    return f"wavefleet://" + json.dumps( kwargs, cls=NumpyEncoder)

def decode_spotter_cache_uri(uri: str):
    uri = uri.replace('wavefleet://', '')
    kwargs = json.loads(uri,object_hook=object_hook)
    request_type = kwargs.pop('request_type')
    return request_type, kwargs

def create_cache(request_type,handler,session):
    if exists():
        cache = filecache._get_cache(CACHE_NAME)
        for remote_resource in cache.resources:
            if isinstance(remote_resource,WaveFleetResource):
                if request_type not in remote_resource._handlers:
                    remote_resource._handlers[request_type] = handler
                break
    else:
        wavefleetresource = WaveFleetResource({request_type:handler},session)
        filecache.create_cache(CACHE_NAME,
                               cache_path=CACHE_PATH,
                               resources=[wavefleetresource],
                               cache_size_GB=2)

def exists():
    return filecache.exists(CACHE_NAME)

def get_data(request_type,spotter_ids:List[str],**kwargs):
    uris = [ spotter_cache_uri(request_type,spotter_id=_id,
                               **kwargs) for _id in spotter_ids ]

    filepaths = filecache.filepaths(uris, cache_name=CACHE_NAME)
    output = [ load(filepath) for filepath in filepaths ]
    return output


def get_data_search(request_type,**kwargs):
    uri = spotter_cache_uri(request_type,**kwargs)
    filepath = filecache.filepaths(uri, cache_name=CACHE_NAME)[0]
    return load(filepath)
