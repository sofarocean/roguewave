"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Module defining the various resources we can use. Here a resource is a class
that acts as a normal file opened for binary read or write in Python. However,
the underlying object may be a an s3 object or a local file. In the case of
an s3 object all IO is buffered and changes occur on s3 only once the file
object is closed.

Classes:
- `Resource`, abstract parent class
- `FileResource`, resource implementation when we use local files (basically
    wrapper around a standard file object)
- `S3Resource`, resource implementation when we use objects stored on S3.

Functions:

- `create_resource`, create a resource, main function to use

How To Use This Module
======================
(See the individual functions for details.)

1. `import create_resource from resources`
2. create a resource: resource = create_resource( uri,mode ), where uri is
    either a local file path, or a s3 uri, and mode is read binary ('rb') or
    write binary ('wb').
"""
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from typing import Union, Sequence, List, Literal
from io import BytesIO
from boto3 import resource
from tqdm import tqdm
from roguewave.wavewatch3.restart_file_cache import get_data


class Resource:
    """
    Context manager base clase defining a resource. needs to be implemented.
    """

    def __init__(
        self,
        resource_location,
        mode: Literal["rb", "wb"] = "rb",
        cache: bool = False,
        cache_name: str = None,
    ):
        """
        :param resource_location:
        :param mode: mode to open file, only binary read ('rb') or binary write
            ('wb') are supported.
        """
        self.resource_location = resource_location
        self.mode = mode
        self.cache = cache
        self.cache_name = cache_name
        self.resource_handle = None

    @property
    def read_only(self):
        """
        :return: resource is opened as readonly
        """
        return self.mode == "rb"

    @property
    def write_only(self):
        """
        :return: resource is opened as writeonly
        """
        return self.mode == "wb"

    def read_range(self, s: Union[slice, Sequence[slice]]) -> List[bytearray]:
        """
        Read a range of bytes as defined by the slice(s). We can
        input a single slice, or multiple slices. We always return a list of
        bytearrays corresponding to the byteranges (list of length 1 if a
        single slice is used as input

        :param s: slice or list of slices
        :return: List of bytearrays
        """

        pass

    def read(self, number_of_bytes=-1) -> bytearray:
        """
        :param number_of_bytes: read number of bytes, if <0 read all bytes to
            end of resource from current posiiton
        :return: return a bytearray of the data read
        """
        pass

    def write(self, _bytes: bytes):
        """
        write a byte array to the resource
        :param _bytes: bytes to write
        :return: None
        """
        pass

    def __enter__(self):
        """
        Dunder method to implement the context management protocol on entry of
        a with block.
        :return:
        """
        return self

    def close(self):
        """
        Close the resource
        :return: None
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Dunder method to implement the context management protocol on entry of
        a with block.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.close()
        self.resource_handle = None

    def __del__(self):
        """
        Dunder method to make sure we release any resources properly.
        :return:
        """
        if self.resource_handle is not None:
            self.close()

    def seek(self, position_bytes):
        """
        Move the current position in the stream to the indicated position.
        :param position_bytes: position in bytes
        :return:
        """
        pass

    def tell(self):
        """
        Return the current position in the stream.
        :return: position in stream in bytes.
        """
        pass


class FileResource(Resource):
    """
    Open a file as a "resource". Only needed becuase we want to abstract away
    the difference between a local and remote file for s3.
    """

    def __init__(
        self,
        resource_location,
        mode: Literal["rb", "wb"] = "rb",
        cache: bool = False,
        cache_name: str = None,
    ):
        """
        :param resource_location:
        :param mode: mode to open file, only binary read ('rb') or binary write
            ('wb') are supported.
        """
        super().__init__(resource_location, mode, False, None)
        self.resource_handle = open(resource_location, mode)

    def read_range(
        self, slices: Union[slice, Sequence[slice]]
    ) -> Union[bytearray, List[bytearray]]:
        """
        Read a range of bytes as defined by the slice(s). We can
        input a single slice, or multiple slices. We always return a list of
        bytearrays corresponding to the byteranges (list of length 1 if a
        single slice is used as input

        :param s: slice or list of slices
        :return: List of bytearrays
        """

        if self.write_only:
            raise IOError(
                "Resource has been opened as write only, cannot" "read from it."
            )

        if isinstance(slices, slice):
            slices = [slices]

        out = []
        for s in slices:
            self.resource_handle.seek(s.start)
            number_bytes = s.stop - s.start
            out.append(bytearray(self.resource_handle.read(number_bytes)))

        return out

    def write(self, _bytes: bytes):
        """
        write a byte array to the resource
        :param _bytes: bytes to write
        :return: None
        """
        self.resource_handle.write(_bytes)

    def read(self, number_of_bytes=-1) -> bytearray:
        """
        :param number_of_bytes: read number of bytes, if <0 read all bytes to
            end of resource from current posiiton
        :return: return a bytearray of the data read
        """
        if self.write_only:
            raise IOError(
                "Resource has been opened as write only, cannot" "read from it."
            )

        return bytearray(self.resource_handle.read(number_of_bytes))

    def close(self):
        """
        Close the resource
        :return: None
        """
        if self.resource_handle is not None:
            self.resource_handle.close()

    def seek(self, position_bytes):
        """
        Move the current position in the stream to the indicated position.
        :param position_bytes: position in bytes
        :return:
        """
        return self.resource_handle.seek(position_bytes)

    def tell(self):
        """
        Return the current position in the stream.
        :return: position in stream in bytes.
        """
        return self.resource_handle.tell()


class S3Resource(Resource):
    """
    Open a s3 object as a "resource". Allows us to interact with an s3 object
    as if it was a regular stream (supports read, write, seek, tell). Obviously
    less performant than local IO.
    """

    boto3_client_lock = Lock()

    def __init__(
        self,
        resource_location,
        mode: Literal["rb", "wb"] = "rb",
        cache: bool = False,
        cache_name: str = None,
    ):
        """
        :param resource_location:
        :param mode: mode to open file, only binary read ('rb') or binary write
            ('wb') are supported.
        """
        super().__init__(resource_location, mode, cache, cache_name)

        bucket, key = resource_location.replace("s3://", "").split("/", maxsplit=1)
        self._key = key
        self._bucket = bucket

        # Creation is not thread safe. Ensure only one thread is creating.
        with self.boto3_client_lock:
            self.resource_handle = resource("s3")
        self._position = 0
        self._write_buffer = BytesIO()
        self._bytes_written = False

    def read_range(self, slices: Union[slice, Sequence[slice]]) -> List[bytearray]:
        """
        Read a range of bytes as defined by the slice(s). We can
        input a single slice, or multiple slices. We always return a list of
        bytearrays corresponding to the byteranges (list of length 1 if a
        single slice is used as input.

        Note that for efficiency we use multiple threads to iterate over the
        byterange. Only the byterange needed is downloaded.

        :param s: slice or list of slices
        :return: List of bytearrays
        """
        if isinstance(slices, slice):
            slices = [slices]

        if self.cache:
            keys = []
            start_byte = []
            stop_byte = []
            for _slice in slices:
                stop_byte.append(_slice.stop)
                start_byte.append(_slice.start)
                keys.append(self.resource_location)
            return get_data(keys, start_byte, stop_byte, self.cache_name)

        if self.write_only:
            raise IOError(
                "Resource has been opened as write only, cannot" "read from it."
            )

        def _worker(s: slice):
            if s.start == 0 and s.stop == -1:
                obj = self.resource_handle.Object(self._bucket, self._key)
                with BytesIO() as file:
                    _ = obj.download_fileobj(file)
                    file.seek(0)
                    _bytes = bytearray(file.read())

                return bytearray(_bytes)
            else:
                obj = self.resource_handle.Object(self._bucket, self._key)
                data = obj.get(Range=f"bytes={s.start}-{s.stop-1}")["Body"]
                return bytearray(data.read())

        if len(slices) > 1:
            with ThreadPool(processes=10) as pool:
                output = list(tqdm(pool.imap(_worker, slices), total=len(slices)))
        else:
            output = [_worker(slices[0])]
        return output

    def read(self, number_of_bytes=-1) -> bytearray:
        """
        :param number_of_bytes: read number of bytes, if <0 read all bytes to
            end of resource from current posiiton
        :return: return a bytearray of the data read
        """
        if self.write_only:
            raise IOError(
                "Resource has been opened as write only, cannot" "read from it."
            )

        start_byte = self._position
        if number_of_bytes < 0:
            end_byte = -1
            _bytes = self.read_range(slice(0, end_byte, 1))[0]
            return _bytes[start_byte:]
        else:
            end_byte = self._position + number_of_bytes
            self._position += number_of_bytes
            return self.read_range(slice(start_byte, end_byte, 1))[0]

    def write(self, _bytes: bytes):
        """
        Write a byte array to the resource. Note that the s3 resource writes
        to an internal buffer. Only when the object is closed is the buffer
        pushed to s3 and "written".
        :param _bytes: bytes to write
        :return: None
        """

        self._write_buffer.write(_bytes)
        self._bytes_written = True

    def _push_to_s3(self):
        """
        Push data in the write buffer to s3.
        :return:
        """

        self._write_buffer.seek(0)
        obj = self.resource_handle.Object(self._bucket, self._key)
        _ = obj.upload_fileobj(self._write_buffer)

    def close(self):
        """
        Close the resource. Flushes the write buffer (if applicable).
        :return: None
        """
        if self._bytes_written:
            self._push_to_s3()
        self._write_buffer.close()
        self._bytes_written = False

    def tell(self):
        """
        Return the current position in the stream.
        :return: position in stream in bytes.
        """
        return self._position

    def seek(self, position_bytes):
        """
        Move the current position in the stream to the indicated position.
        :param position_bytes: position in bytes
        :return:
        """
        self._position = position_bytes


def create_resource(
    uri: str, mode: Literal["wb", "rb"] = "rb", cache=False, cache_name=None
) -> Resource:
    """
    Create the appropriate resource object basded on the "uri". If the uri
    starts with "s3://" we assume it refer to a s3 object in the form
    "s3://bucket/key". Otherwise we assume it is the path to a local file.
    :param uri: s3 uri or path to local file.
    :param mode: Whether to open resource in binary read ('rb') or binary write
        mode.
    :return: resource object
    """
    if "s3://" in uri:
        return S3Resource(
            resource_location=uri, mode=mode, cache=cache, cache_name=cache_name
        )
    else:
        return FileResource(
            resource_location=uri, mode=mode, cache=cache, cache_name=cache_name
        )
