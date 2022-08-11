import numpy
from roguewave.wavewatch3.io import FortranInt,FortranFloat,FortranCharacter, \
    Resource
from io import BytesIO
from dataclasses import dataclass

_ENDIANNESS = '<'

@dataclass()
class Grid:
    number_of_spatial_points: int
    frequencies: numpy.ndarray
    directions: numpy.ndarray
    latitude: numpy.ndarray
    longitude: numpy.ndarray
    to_linear_index: numpy.ndarray
    to_point_index: numpy.ndarray

    def extract(self, s:slice, var:numpy.ndarray):
        linear_indices = numpy.arange(s.start, s.stop, s.step,dtype='int32')
        ix = self.to_point_index[ 0 , linear_indices]
        iy = self.to_point_index[ 1 , linear_indices]
        return var[ix,iy]

    def project( self, sx:slice, sy:slice, var:numpy.ndarray, except_val=numpy.nan ):
        ix = numpy.arange(sx.start, sx.stop, sx.step, dtype='int32')
        iy = numpy.arange(sy.start, sy.stop, sy.step, dtype='int32')
        ind = self.to_linear_index[ix,iy]
        mask = ind >=0
        ind = ind[ mask ]
        ix = ix[mask]
        iy = iy[mask]

        nx = len(ix)
        ny = len(iy)
        out = numpy.zeros( (nx,ny), dtype=var.dtype ) + except_val
        out[ix,iy] = var[ind]
        return out

def read_model_definition(reader:Resource):

    def jump( stream, number,start ):
        if start is None:
            _ = stream.read(number)
        else:
            stream.seek(start+number)

    stream = BytesIO( reader.read() )

    fort_char = FortranCharacter(endianness=_ENDIANNESS)
    fort_int = FortranInt(endianness=_ENDIANNESS)
    fort_float = FortranFloat(endianness=_ENDIANNESS)

    data = {}
    start_of_record_absolute = 0
    _sor = fort_int.unpack(stream,1)[0] # Fortran record opening
    data['id'] = fort_char.unpack(stream,35) # Line 70
    data['grid_name'] = fort_char.unpack(stream,10) # Line 70  # noqa: F841
    data['nx'] = fort_int.unpack(stream,1)[0] # Number of longitude points
    data['ny'] = fort_int.unpack(stream, 1)[0] # Number of longitude points
    data['num_sea'] = fort_int.unpack(stream, 1)[0] # Number of sea points
    data['num_dir'] = fort_int.unpack(stream, 1)[0] # Number of directional points
    data['num_freq'] = fort_int.unpack(stream, 1)[0] # Number of wavenumber points

    # Number of input bound points (see w3odatd.ftn, line 219) and
    # Number of files for output bound data (see w3odatd.ftn, line 220)
    # (3 int total, not used here, skipping, jumping to eor
    _ = jump( stream, _sor + 4, 0 )
    _eor = fort_int.unpack(stream,1)[0]
    assert _sor == _eor

    _sor = fort_int.unpack(stream, 1)[0]
    _ = jump(stream, _sor, None)
    _eor = fort_int.unpack(stream,1)[0]
    assert _sor == _eor

    #assert fort_int.unpack(stream,1)[0] == _sor

    # Enter the section for W3GDAT (line 582)
    data["grid_type"], data["flagll"],data["iclose"] = \
        fort_int.unpack(stream,None,unformatted_sequential=True)
    ( data["longitude_stepsize"], data["latitude_stepsize"],
      data["longitude_start"],data["latitude_start"] ) = \
        fort_float.unpack(stream, None, unformatted_sequential=True)

    _sor = fort_int.unpack(stream, 1)[0]
    start_of_record_absolute = stream.tell()
    dtype_float = numpy.dtype('float32').newbyteorder("<")
    dtype_int = numpy.dtype('int32').newbyteorder("<")

    # First the bottom grid. The bottom grid is only stored on the computational
    # grid
    data['bottom_datum'] = \
        numpy.frombuffer(
            stream.read(data['num_sea']*4),
            count=data['num_sea'], dtype=dtype_float)

    # Next the mask layer. The masked layer is stored everywhere
    num_points = data['nx'] * data['ny']
    data['mask'] = \
        numpy.reshape(
            numpy.frombuffer(stream.read(num_points*4),dtype=dtype_int),
            ( data['nx'] , data['ny'] )
        )

    # Next the mapping that relates ix,iy -> ns. Subtract 1 to account for 0
    # based indexing (vs 1 in Fortran).
    data['to_linear_index'] = \
        numpy.reshape(
            numpy.frombuffer(stream.read(num_points*4),dtype=dtype_int),
            ( data['nx'] , data['ny'] )
        ) - 1

    num_points = data['num_sea']
    data['to_point_index'] =     data['to_linear_index'] = \
        numpy.reshape(
            numpy.frombuffer(stream.read(num_points*8),dtype=dtype_int),
            ( 2 , num_points )
        ) - 1

    # FROM: w3gdatmd.ftn line 160
    #      TRFLAG    Int.  Public   Flag for use of transparencies
    #                                0: No sub-grid obstacles.
    #                                1: Obstructions at cell boundaries.
    #                                2: Obstructions at cell centers.
    #                                3: Like 1 with continuous ice.
    #                                4: Like 2 with continuous ice.
    data['tr_flag'] = fort_int.unpack(stream,1)[0]
    _ = jump(stream,_sor,start_of_record_absolute)
    _eor = fort_int.unpack(stream,1)[0]
    assert _sor == _eor

    # we jump three values associated with TRFLAG
    for ii in range(0,3):
        _ = fort_int.unpack(stream,None,True)

    # Spectral parameters
    # For descriptions; see w3gdatmd.ftn
    num_spec_points = data['num_freq'] * data['num_dir']


    # I do not fully understand why arrays we ignore are larger.. but we
    # ignore them
    byte_size_full = (num_spec_points + data['num_dir']) * 4

    # ignore: "mapwn, mapth"
    _sor = fort_int.unpack(stream, 1)[0]
    _ = stream.read(byte_size_full * 2)
    data['direction_step_size'] = fort_float.unpack(stream, 1)[0]
    data['direction_degree'] = numpy.frombuffer(
        stream.read( data['num_dir'] * 4 ), dtype=dtype_float
    ) * 180/numpy.pi

    # ignore: "stuff"
    _ = stream.read(byte_size_full * 5)

    #
    data['freq_mult_fac'] = fort_float.unpack(stream,1)[0]
    data['start_frequency'] = fort_float.unpack(stream, 1)[0]

    # Note they store an extra frequency before start, and at the end, I
    # assume this is for convinience in calculating delta's- either way,
    # ignored here.
    data['frequency_hertz'] = ( numpy.frombuffer(
        stream.read( data['num_freq'] * 4 + 8 ), dtype=dtype_float
    ) /numpy.pi/2 )[1:-1]

    latitude = numpy.linspace(
            data['latitude_start'],
            data['latitude_start'] + data['latitude_stepsize']*data['ny'],
            endpoint=True
    )

    longitude = numpy.linspace(
            data['longitude_start'],
            data['longitude_start'] + data['longitude_stepsize']*data['nx'],
            endpoint=True
    )
    grid = Grid( number_of_spatial_points=data['num_sea'],
                 frequencies=data['frequency_hertz'],
                 directions=data['direction_degree'],
                 latitude=latitude,
                 longitude=longitude,
                 to_linear_index=data['to_linear_index'],
                 to_point_index=data['to_point_index'])

    return grid, -data['bottom_datum'], data['mask']

