from multiprocessing.pool import ThreadPool
from typing import Union, Sequence, List, Literal
from io import BytesIO
from boto3 import resource
from tqdm import tqdm
import tempfile


class Resource():
    def __init__(self, resource_location, mode:Literal['rb','wb']='rb'):
        self.resource_location = resource_location
        self.mode = mode

    @property
    def read_only(self):
        return self.mode == 'rb'

    @property
    def write_only(self):
        return self.mode == 'wb'

    def read_range(self, s:Union[slice,Sequence[slice]]
                   ) -> List[bytearray]:
        pass

    def read(self, number_of_bytes=-1) -> bytearray:
        pass

    def write(self, _bytes:bytes):
        pass

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.resource_handle = None

    def __del__(self):
        if self.resource_handle is not None:
            self.close()

    def seek(self, position_bytes):
        pass

    def tell(self):
        pass


class FileResource(Resource):
    def __init__(self, resource_location, mode:Literal['rb','wb']='rb'):
        super().__init__(resource_location, mode)
        self.resource_handle = open(resource_location, mode)

    def read_range(self, slices:Union[slice,Sequence[slice]]) \
            -> Union[bytearray,List[bytearray]]:

        if self.write_only:
            raise IOError('Resource has been opened as write only, cannot'
                             'read from it.')

        if isinstance(slices,slice):
            slices = [slices]

        out = []
        for s in slices:
            self.resource_handle.seek(s.start)
            number_bytes = s.stop - s.start
            out.append(bytearray(self.resource_handle.read(number_bytes)))

        return out

    def write(self,_bytes:bytes):
        self.resource_handle.write(_bytes)

    def read(self, number_of_bytes=-1) -> bytearray:
        if self.write_only:
            raise IOError('Resource has been opened as write only, cannot'
                             'read from it.')

        return bytearray(self.resource_handle.read(number_of_bytes))

    def close(self):
        if self.resource_handle is not None:
            self.resource_handle.close()

    def seek(self, position_bytes):
        return self.resource_handle.seek(position_bytes)

    def tell(self):
        return self.resource_handle.tell()


class S3Resource(Resource):
    def __init__(self, resource_location, mode:Literal['rb','wb']='rb'):
        super().__init__(resource_location,mode)

        bucket, key = \
            resource_location.replace('s3://', '').split('/', maxsplit=1)
        self._key = key
        self._bucket = bucket
        self.resource_handle = resource('s3')
        self._position = 0
        self._write_buffer = BytesIO()
        self._bytes_written = False


    def read_range(self, slices: Union[slice, Sequence[slice]]) \
            ->List[bytearray]:

        if self.write_only:
            raise IOError('Resource has been opened as write only, cannot'
                             'read from it.')

        def _worker( s:slice):
            obj = self.resource_handle.Object(self._bucket, self._key)
            data = obj.get(Range=f'bytes={s.start}-{s.stop-1}')['Body']
            return bytearray(data.read())

        if isinstance(slices,slice):
            slices = [slices]

        if len(slices) > 1:
            with ThreadPool(processes=10) as pool:
                output = list(
                    tqdm(
                        pool.imap(_worker, slices),
                        total=len(slices)
                    )
                )
        else:
            output = [_worker(slices[0])]
        return output

    def _read_all(self) -> bytearray:
        if self.write_only:
            raise IOError('Resource has been opened as write only, cannot'
                             'read from it.')

        obj = self.resource_handle.Object(self._bucket, self._key)
        with BytesIO() as file:
            response = obj.download_fileobj(file)
            file.seek(0)
            _bytes = bytearray(file.read())

        return _bytes

    def read(self, number_of_bytes=-1) -> bytearray:

        if self.write_only:
            raise IOError('Resource has been opened as write only, cannot'
                             'read from it.')

        start_byte = self._position

        if number_of_bytes < 0:
            _bytes = self._read_all()
            return _bytes[start_byte:]
        else:
            end_byte = self._position + number_of_bytes
            self._position += number_of_bytes
            return self.read_range(slice( start_byte,end_byte,1 ))[0]

    def write(self, _bytes:bytes):
        self._write_buffer.write(_bytes)
        self._bytes_written = True

    def _push_to_s3(self):
        self._write_buffer.seek(0)
        obj = self.resource_handle.Object(self._bucket, self._key)
        response = obj.upload_fileobj(self._write_buffer)

    def close(self):
        if self._bytes_written:
            self._push_to_s3()
        self._write_buffer.close()
        self._bytes_written = False

    def tell(self):
        return self._position

    def seek(self, position_bytes):
        self._position = position_bytes


def create_resource(uri:str,mode:Literal['wb','rb']='rb'):
    if 's3://' in uri:
        return S3Resource(resource_location=uri,mode=mode)
    else:
        return FileResource(resource_location=uri,mode=mode)