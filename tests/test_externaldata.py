from datetime import datetime,timezone
from pysofar.spotter import Spotter
from roguewave.externaldata.spotter import get_spectrum_from_sofar_spotter_api

# Spotter ID
spotter_id = 'SPOT-0740'

# Start time
start_date = datetime(2021,1,1,0,0,0,tzinfo=timezone.utc)

# End time
end_date = datetime(2021,1,6,0,0,0,tzinfo=timezone.utc)

# exact date
exact_first_date = datetime(2021,1,1,0,39,31,tzinfo=timezone.utc)

limit = 21

def test_get_spectrum_from_sofar_spotter_api():
    spotter = Spotter(spotter_id, spotter_id)
    spectra = get_spectrum_from_sofar_spotter_api(spotter, start_date, end_date,limit=limit)

    assert len(spectra) == limit
    assert spotter.id == spotter_id
    assert spectra[0].timestamp >= start_date
    assert spectra[0].timestamp == exact_first_date
    assert spectra[-1].timestamp < end_date
    assert spectra[0].hm0() == 3.2687019746139963