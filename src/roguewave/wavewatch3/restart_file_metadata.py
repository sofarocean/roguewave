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
- `MetaData`, dataclass containing metadata from a restart file

Functions:

- `read_header`, read the header information from a restart file opened by the
    given resource.


How To Use This Module
======================
(See the individual functions for details.)

1. import
2. read metadat using `read_header` which returns a MetaData object.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Literal

from roguewave.wavewatch3.fortran_types import FortranCharacter, FortranInt
from roguewave.wavewatch3.resources import Resource


@dataclass()
class MetaData:
    """
    Class containing all the metadata we need that is stored in a ww3 restart
    file.
    """
    name: str
    version: "str"
    grid_name: "str"
    restart_type: "str"
    nsea: int
    nspec: int
    record_size_bytes: int
    time: datetime
    byte_order: Literal["<", ">", "="]
    float_size: int


def read_header(resource: Resource,
                number_of_spectral_points = 36 * 36,
                byte_order = '<', float_size = 4) -> MetaData:
    """
    Function to read header information in a restart file and return the
    needed MetaData object.

    :param resource: an instance of a resource.
    :param number_of_spectral_points: number of spectral points. This may not
        be known in advance- but due to the way the file is layed out a guess
        that is larger than 78 bytes will work
    :param byte_order: Sofar ww3 uses little endian
    :param float_size: 4 bytes for Sofar (float32)
    :return:
    """
    data = {}

    # first read the character arrays for name, version, grid name and
    # restart_type. We do not know the record size yet with which the file
    # was written, so we just guess a record size.
    guess_record_size_bytes = number_of_spectral_points * float_size
    stream = BytesIO(resource.read(guess_record_size_bytes * 2))

    fort_char = FortranCharacter(endianness=byte_order)
    fort_int = FortranInt(endianness=byte_order)

    data['byte_order'] = byte_order
    data['float_size'] = float_size
    data["name"] = fort_char.unpack(stream, 26)
    data["version"] = fort_char.unpack(stream, 10)
    data["grid_name"] = fort_char.unpack(stream, 30)
    data["restart_type"] = fort_char.unpack(stream, 4)

    # Now we can read the number of spatial points and number of spectral points
    data["nsea"] = fort_int.unpack(stream, 1)[0]
    data["nspec"] = fort_int.unpack(stream, 1)[0]
    data['record_size_bytes'] = \
        data["nspec"] * float_size

    if (guess_record_size_bytes*2 < data['record_size_bytes'] + 8):
        # We only need to read 8 more bytes than the actual length to get the
        # time information. Now we know the actual record length, see if we
        # can just go ahead, or if we need to reload data from the resource.
        # Especially if the restart file is remote this saves us another
        # request.
        stream.seek(0)
        stream =   BytesIO(resource.read(data['record_size_bytes'] + 8))

    # Jump to the second record which contains time information
    stream.seek(data['record_size_bytes'])

    date_int = fort_int.unpack(stream, 1)[0]
    year, month, day = _unpack_date_time_from_int(date_int)

    time_int = fort_int.unpack(stream, 1)[0]
    hour, min, sec = _unpack_date_time_from_int(time_int)

    data["time"] = datetime(year, month, day, hour, min, sec,
                            tzinfo=timezone.utc)
    return MetaData(**data)


def _unpack_date_time_from_int(t):
    """
    Wavewatch 3 restart files store dates and times as integers in the form:
        20220101 for '2022-01-01' or 193000 for '19:30:00Z'. This is a simple
        function to unpack either the year/month/day ore hour/minute/second
        triple.
    :param t: integer containing date or time
    :return: time or date tripple.
    """
    x = int(t / 10000)
    y = int((t - x * 10000) / 100)
    z = t - x * 10000 - 100 * y
    return x,y,z