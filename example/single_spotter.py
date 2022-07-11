from roguewave import load_spectrum, convert_to_2d_spectrum, get_spectral_partitions_from_2dspectra, get_bulk_partitions_from_spectral_partitions, set_log_to_console
from datetime import timedelta
import matplotlib.pyplot as plt
spectra1d = load_spectrum('single_spotter.pkl')
set_log_to_console()
spec2d = convert_to_2d_spectrum( spectra1d, number_of_directions=36 )
spec2d_highres = convert_to_2d_spectrum( spectra1d, number_of_directions=144)


part = get_spectral_partitions_from_2dspectra(spec2d,timedelta(hours=12))
part_highres = get_spectral_partitions_from_2dspectra(spec2d_highres,timedelta(hours=12))


bulk = get_bulk_partitions_from_spectral_partitions(part)
bulk_highres = get_bulk_partitions_from_spectral_partitions(part_highres)


def plot(bulk,col):
    for spotterid,partitions in bulk.items():
        for partition in partitions:
            plt.plot( partition['timestamp'],partition['peak_spread'] )
            plt.xticks(rotation=45)

#plot(bulk_highres,'r')
plot(bulk,'k')
plt.figure()
plot(bulk_highres,'r')
plt.show()





