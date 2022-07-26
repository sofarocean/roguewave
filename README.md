# Sofar Ocean API Python Client
Python toolbox to interact with Sofar Ocean Spotter API.

### Requirements
- Python3
- pysofar
- numpy
- netCDF4
- wheel (If developing/Contributing)
- Pytest (If developing/Contributing)
- Setuptools (If developing/Contributing)

### Installation
1. Make sure that you have the requirements listed above
2. `pip install roguewave` to your desired python environment
3. Test with `python3 -c 'import roguewave'`. If this runs successfully, chances are everything worked.
4. To interact with the api's you need to add the access tokens into an environmental file. 
   Specifically create a file called `sofar_api.env` in your user directory containing
```shell
WF_API_TOKEN=_wavefleet_token_
SPECTRAL_API_TOKEN=_spectral_api_token_
```



# Examples
### Retrieve Spectrum from observational Spotter API
```python
from roguewave.spotterapi import get_spectrum
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# The Spotter we want the data from
spotter_id = 'SPOT-0740'

# Start time to grab
start_date = datetime(2021,1,1,0,0,0,tzinfo=timezone.utc)

# End time
end_date = datetime(2021,1,6,0,0,0,tzinfo=timezone.utc)

# You will need a valid access token setup for the pysofar library for this to
# work. We refer to the pysofar documentation how to set that up.


# Get the spectra
spectra = get_spectrum(spotter_id,start_date,end_date)

# We now have spectra from the spotter we can interact with
for key in spectra:
    for spectrum in spectra[key]:
        string = f'The waveheight at {spectrum.timestamp} was {spectrum.hm0()} meter'
        print(string)

# or do a simple plot
plt.plot(spectra[spotter_id][0].frequency, spectra[spotter_id][0].variance_density,'k')
plt.xlabel('Frequency (hz)')
plt.ylabel('Variance Density (m$^2$/s)')
plt.yscale('log')
plt.title(f'spectrum for {spotter_id} at time {spectra[spotter_id][0].timestamp}')
plt.grid()
plt.show()
```

### Retrieve Spectrum from model Spectral API

```python
from roguewave.modeldata.sofarspectralapi import SofarSpectralAPI,load_sofar_spectral_file
import matplotlib.pyplot as plt

# Create an api Object
api = SofarSpectralAPI()

# Get all the points accessible for this particular user. This returns a list
# dictionaries containing latitudes and longitudes
points = api.points()

# Lets get the first point. This will download the netcdf containing the spectral
# forcast into the given directory.
file = api.download_spectral_file(**points[0], directory='./')

# Lets load the point. This will open the Netcdf file and return a Spectrum2D
# object. This object supports all the methods of the Spectrum1D object (hm0, 
# tm02, etc.)
data = load_sofar_spectral_file(file)

# Lets plot it
plt.pcolormesh(data[0].frequency, data[0].direction, data[0].variance_density)
plt.show()
```