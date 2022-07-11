from roguewave import convert_to_2d_spectrum, load_spectrum,save_spectrum
from roguewave.wavespectra.partitioning.partitioning import partition_spectrum
import timeit

spectra1D = load_spectrum('spectrum1d.json')
spectra2D = convert_to_2d_spectrum(spectra1D)



references = load_spectrum('reference.zip')

def test():
    partitions, _ = partition_spectrum(spectra2D[0])
    return partitions








if __name__ == '__main__':
    for ii in range(0,1):
        partitions = test()

    for partition, reference in zip(partitions, references):
        par = partitions[partition]
        ref = references[reference]
        dif = par.m0() - ref.m0()
        assert dif ==0


    time = timeit.timeit("test()", setup="from __main__ import test", number=100)
    print( f'time: {time},  reference time: 0.4630766900000003' )