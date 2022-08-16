from tests.restart_files import REMOTE_HASH, REMOTE_RESTART_FILE, \
    REMOTE_MOD_DEF, TEST_DIR, LOCAL_FILE_NAME, REMOTE_WRITE_PATH, file_hash, \
    clone_remote
from roguewave import RestartFile,open_restart_file
import numpy

def test_m0():
    restart_file = clone_remote()
    m0 = restart_file.variance(slice(None),slice(None))
    hm0 = 4 * numpy.sqrt(m0)

    #-25.5, -44.0


    import matplotlib.pyplot as plt
    plt.pcolormesh( restart_file.longitude,restart_file.latitude,hm0 )
    plt.colorbar()
    plt.show()


def subsea7_stuff():
    restart_file = clone_remote()
    m0 = restart_file.variance(slice(None), slice(None))
    hm0 = 4 * numpy.sqrt(m0)

    ilat = int((-25.5+90)/0.5)
    print(restart_file.latitude[ilat])
    ilon = int((-44)%360/0.5)
    print(restart_file.latitude[ilat],restart_file.longitude[ilon])
    print(hm0[ilat,ilon])
    # -25.5, -44.0

if __name__ == '__main__':
    subsea7_stuff()