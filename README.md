# Sofar Ocean API Python Client
Python toolbox to interact with Sofar Ocean Spotter API.

### Requirements
- Python3
- pysofar
- numpy
- Pytest (If developing/Contributing)
- Setuptools (If developing/Contributing)

### Installation
1. Make sure that you have the requirements listed above
2. `pip install roguewave` to your desired python environment
3. Test with `python3 -c 'import roguewave'`. If this runs successfully, chances are everything worked.

# Examples

### Retrieve Spectrum from API
```python
from roguewave import get_spectrum_from_sofar_spotter_api
from pysofar.spotter import Spotter
from datetime import datetime, timezone

# The Spotter we want the data from
spotter_id = 'SPOT-0740'

# Start time to grab
start_date = datetime(2021,1,1,0,0,0,tzinfo=timezone.utc)

# End time
end_date = datetime(2021,1,6,0,0,0,tzinfo=timezone.utc)

# You will need a valid access token setup for the pysofar library for this to
# work. We refer to the pysofar documentation how to set that up.
spotter = Spotter(spotter_id,spotter_id)

# Get the spectra
spectra = get_spectrum_from_sofar_spotter_api(spotter,start_date,end_date,limit=10)

# We now have spectra from the spotter we can interact with
for spectrum in spectra:
    string = f'The waveheight at {spectrum.timestamp} was {spectrum.hm0()} meter'
    print(string)

# or do a simple plot
plt.plot(spectra[0].frequency, spectra[0].variance_density,'k')
plt.xlabel('Frequency (hz)')
plt.ylabel('Variance Density (m$^2$/s)')
plt.yscale('log')
plt.title(f'spectrum for {spotter_id} at time {spectra[0].timestamp}')
plt.grid()
plt.show()
