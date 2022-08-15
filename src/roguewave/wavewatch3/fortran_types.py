import struct
from io import BufferedIOBase


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
