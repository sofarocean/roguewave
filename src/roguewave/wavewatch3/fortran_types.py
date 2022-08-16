"""
Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
=========================
Module to define the types used to write read data from Fortran generated.
This is an internal module and there should not be a reason to use this
directly.

Classes:
- `FortranType`, abstract parent class
- `FortranFloat`, class to read Fortran 4 byte floats (float32)
- `FortranInt`, class to read Fortran 4 byte integers (int32)
- `FortranCharacters`, class to read a Fortran character array and return a
    normal python string.

Functions:

- `N/A`

How To Use This Module
=========================
(See the individual functions for details.)

1. Don't :-)
"""
import struct
from io import BufferedIOBase
from typing import Tuple


class FortranType:
    """
    Base class.
    """

    kind = ''  # kind of variable- kind strings are those used by the struct
    # library
    byte_length_type = 1  # bytelength of an individual instance of the data

    def __init__(self, endianness="<"):
        self.endianness = endianness

    def byte_length(self, number: int) -> int:
        """
        bytelength of multiple instances of the data
        :param number: number of type instances
        :return: total lenght in bytes
        """
        return number * self.byte_length_type

    def format(self, number: int) -> str:
        """
        :param number: number of type instanes to read
        :return: format to use with struct unpack to read the fortran data.
        """
        return f"{self.endianness}{self.kind * number}"

    def _read(self, number: int, stream: BufferedIOBase,
              unformatted_sequential=False) -> Tuple[str, bytes]:
        """


        :param number: number of type instanes to read
        :param stream: byte stream
        :param unformatted_sequential: if this is an unformatted sequential
            read
        :return: format to use in unpack and the data (bytes)

        Note on sequential unformatted data:
        https://gcc.gnu.org/onlinedocs/gfortran/
            File-format-of-unformatted-sequential-files.html

        Read sequential unformatted data written by a Fortran program compiled
        with the GFortran compiler. Each record in the file contains a leading
        start of record (sor) 4 byte marker indicating size N of the record,
        the data written (N bytes), and a trailing end of record (eor) marker
        again indicating the size of the data. No information on the type of
        the data is provided, and this has to be known in advance.

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

    def _unpack(self, number: int, stream: BufferedIOBase,
                unformatted_sequential=False):
        """
        Unpack data
        :param number: number of types to unpack
        :param stream: byte stream
        :param unformatted_sequential: if this is an unformatted sequential
            read
        :return: unpacked data
        """
        return struct.unpack(
            *self._read(number, stream, unformatted_sequential))

    def unpack(self, stream: BufferedIOBase, number=1,
               unformatted_sequential=False):
        """
        Wrapper function. Overwritten by child classes in case some additional
        processing neads to be done before returning the data (e.g. convert
        chart array to str).
        :param stream: byte stream
        :param number: number of types to unpack
        :param unformatted_sequential: if this is an unformatted sequential
            read
        :return: unpacked fortran data as corresponding python type.
        """

        return self._unpack(number, stream, unformatted_sequential)

    def _read_record_marker(self, stream: BufferedIOBase) -> int:
        """
        Read a record marker for an unformatted sequential file.
        :param stream: byte stream
        :return: size of record
        """
        return struct.unpack(
            f"{self.endianness}i", stream.read(4)
        )[0]


class FortranCharacter(FortranType):
    """
    Class to read fortran character types and convert to string.
    """
    kind = 'B'

    def unpack(self, stream: BufferedIOBase, number=1,
               unformatted_sequential=False):
        return "".join([chr(x) for x in
                        self._unpack(number, stream, unformatted_sequential)])


class FortranFloat(FortranType):
    """
    Class to read fortran 32 bit float types and convert to string.
    """
    kind = 'f'
    byte_length_type = 4


class FortranInt(FortranType):
    """
    Class to read fortran 32 bit integer types and convert to string.
    """
    kind = 'i'
    byte_length_type = 4
