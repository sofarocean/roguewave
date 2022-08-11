import struct
from io import BytesIO, BufferedIOBase
from typing import Union, Sequence, List
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

from boto3 import resource


class FortranType:
    kind = ''
    byte_length_type = 1

    def __init__(self, endianness="<"):
        self.endianness = endianness

    def byte_length(self, number):
        return number * self.byte_length_type

    def format(self, number):
        return f"{self.endianness}{self.kind * number}"

    def _read(self, number, stream: BufferedIOBase,
              unformatted_sequential=False):
        """
        https://gcc.gnu.org/onlinedocs/gfortran/File-format-of-unformatted-sequential-files.html

        Read sequential unformatted data written by a Fortran program compiled
        with the GFortran compiler. Each record in the file contains a leading
        start of record (sor) 4 byte marker indicating size N of the record,
        the data written (N bytes), and a trailing end of record (eor) marker again
        indicating the size of the data. No information on the type of the data is
        provided, and this has to be known in advance.

        :return:
        """
        if unformatted_sequential:
            sor = self._read_record_marker(stream)
            number = sor // self.byte_length_type

        data = stream.read(self.byte_length(number))
        fmt = self.format(number)

        if unformatted_sequential:
            eor = self._read_record_marker(stream)
            assert sor == eor
        return fmt, data

    def _unpack(self, number, stream: BufferedIOBase,
                unformatted_sequential=False):
        return struct.unpack(
            *self._read(number, stream, unformatted_sequential))

    def unpack(self, stream: BufferedIOBase, number=1,
               unformatted_sequential=False):
        return self._unpack(number, stream, unformatted_sequential)

    def _read_record_marker(self, stream):
        return struct.unpack(
            f"{self.endianness}i",stream.read(4)
        )[0]


class FortranCharacter(FortranType):
    kind = 'B'

    def unpack(self, stream: BufferedIOBase, number=1,
               unformatted_sequential=False):
        return "".join([chr(x) for x in
                        self._unpack(number, stream,unformatted_sequential)])


class FortranFloat(FortranType):
    kind = 'f'
    byte_length_type = 4

class FortranInt(FortranType):
    kind = 'i'
    byte_length_type = 4

class FortranRecordMarker(FortranType):
    kind = 'i'
    byte_length_type = 4

    def unpack(self, stream: BufferedIOBase, number=1,
               unformatted_sequential=False):
        return self._unpack(number, stream)[0]


class Resource():
    def read_range(self, s:Union[slice,Sequence[slice]]) -> bytes:
        pass

    def read(self, number_of_bytes=-1) -> bytes:
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
    def __init__(self, file_name):
        self.resource_handle = open(file_name, 'rb')

    def read_range(self, slices:Union[slice,Sequence[slice]]) \
            -> Union[bytes,List[bytes]]:

        if isinstance(slices,slice):
            slices = [slices]

        out = []
        for s in slices:
            self.resource_handle.seek(s.start)
            number_bytes = s.stop - s.start
            out.append(self.resource_handle.read(number_bytes))
        return out

    def read(self, number_of_bytes=-1) -> bytes:
        return self.resource_handle.read(number_of_bytes)

    def close(self):
        if self.resource_handle is not None:
            self.resource_handle.close()

    def seek(self, position_bytes):
        return self.resource_handle.seek(position_bytes)

    def tell(self):
        return self.resource_handle.tell()


class S3Resource(Resource):
    def __init__(self, s3_uri):
        bucket, key = \
            s3_uri.replace('s3://', '').split('/', maxsplit=1)
        self._key = key
        self._bucket = bucket
        self.resource_handle = resource('s3')
        self._position = 0

    def read_range(self, slices: Union[slice, Sequence[slice]]) \
            -> Union[bytes, List[bytes]]:

        def _worker( s:slice):
            obj = self.resource_handle.Object(self._bucket, self._key)
            data = obj.get(Range=f'bytes={s.start}-{s.stop-1}')['Body']
            return data.read()

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

    def _read_all(self):
        obj = self.resource_handle.Object(self._bucket, self._key)
        return obj.get()['Body'].read()

    def read(self, number_of_bytes=-1) -> bytes:
        if number_of_bytes == -1:
            return self._read_all()

        start_byte = self._position
        end_byte = self._position + number_of_bytes
        self._position += number_of_bytes
        return self.read_range(slice( start_byte,end_byte,1 ))[0]

    def close(self):
        self.resource_handle = None

    def tell(self):
        return self._position

    def seek(self, position_bytes):
        self._position = position_bytes


def create_resource(uri:str):
    if 's3://' in uri:
        return S3Resource(s3_uri=uri)
    else:
        return FileResource(file_name=uri)