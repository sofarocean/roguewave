from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Literal

from roguewave.wavewatch3.fortran_types import FortranCharacter, FortranInt
from roguewave.wavewatch3.resources import Resource


@dataclass()
class MetaData:
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


def unpack_date_time_from_int( t ):
    x = int(t / 10000)
    y = int((t - x * 10000) / 100)
    z = t - x * 10000 - 100 * y
    return x,y,z


def read_header(reader: Resource,
                guess_number_of_spectral_points = 36 * 36,
                byte_order = '<', float_size = 4) -> MetaData:
    data = {}

    # first read the character arrays for name, version, grid name and
    # restart_type. We do not know the record size yet with which the file
    # was written, so we just guess a record size.
    guess_record_size_bytes = guess_number_of_spectral_points * float_size
    stream = BytesIO( reader.read(guess_record_size_bytes*2) )

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
        stream =   BytesIO( reader.read(data['record_size_bytes'] + 8) )

    # Jump to the second record which contains time information
    stream.seek(data['record_size_bytes'])

    date_int = fort_int.unpack(stream, 1)[0]
    year, month, day = unpack_date_time_from_int( date_int )

    time_int = fort_int.unpack(stream, 1)[0]
    hour, min, sec = unpack_date_time_from_int(time_int)

    data["time"] = datetime(year, month, day, hour, min, sec,
                            tzinfo=timezone.utc)
    return MetaData(**data)