"""
Note for this script to work pysofar needs to be configured with a token that has access to 'SPOT-0661'.
"""

from src.roguewave import get_spotter_data, FrequencySpectrum
from src.roguewave.wavephysics.windestimate import estimate_u10_from_spectrum
from src.roguewave.spotter.read_csv_data import read_spectra
from pandas import DataFrame, concat, to_datetime
from datetime import datetime, timedelta
import numpy
import matplotlib.pyplot as plt
import argparse
import os
import requests
import pandas as pd
import json
from numpy import linspace, errstate, sqrt, interp, full_like, inf, cos, sin, pi, nan
from xarray import Dataset
from roguewave.tools.time import datetime64_to_timestamp
import numpy as np
import pandas
from xarray import DataArray
from datetime import timezone
from roguewave import load
from roguewave.spotter.analysis import spotter_frequency_response_correction

def get_ndbc():
    stations = {
        "46026":'/Users/pietersmit/Downloads/46026.csv',
    }

    data = {}
    for station in stations:
        dat = pandas.read_csv( stations[station],sep=' ',skipinitialspace=True,skiprows=[1])
        time_heading = ['#YY', 'MM', 'DD', 'hh', 'mm']
        time_list = [ dat[x].values for x in time_heading  ]
        time = []
        for y,m,d,h,min in zip( *time_list):
            time.append( datetime(int(y),int(m),int(d),int(h),int(m),tzinfo=timezone.utc) )

        val = pandas.to_numeric(dat['WSPD'],errors='coerce')
        val2 = pandas.to_numeric(dat['WVHT'],errors='coerce')
        data[station] = DataArray( data=val, dims='time',coords={'time':time} )
        wh = DataArray( data=val2, dims='time',coords={'time':time} )
    data = data['46026']
    return data, wh


def pullHdrDataIntoRougeWave(json_data):
    frequencies = (
        linspace(
            3,
            82,
            79,
            endpoint=False,
        )
        * 0.009765625
    )
    time = []
    Szz = []
    a1 = []
    b1 = []
    a2 = []
    b2 = []
    for item in json_data:
        ended_at = item['results']['metadata']['endedAt']
        norm_factor = int(item['results']['normFactor'])
        szz = []
        aa1 = []
        bb1 = []
        aa2 = []
        bb2 = []
        time.append(float(ended_at))
        for sample in item['results']['full']:
            aa1.append(sample['a1'])
            bb1.append(sample['b1'])
            aa2.append(sample['a2'])
            bb2.append(sample['b2'])
            E = sample['Szz'] * norm_factor / (2.5/256) / 1000000
            szz.append(E)
        Szz.append(szz)
        a1.append(aa1)
        b1.append(bb1)
        a2.append(aa2)
        b2.append(bb2)
    time = np.array(time)
    Szz = np.array(Szz)
    a1 = np.array(a1)
    b1 = np.array(b1)
    a2 = np.array(a2)
    b2 = np.array(b2)

    print(time.shape)
    print(frequencies.shape)
    print(Szz.shape)
    time = to_datetime(time, unit="s")
    latitude = full_like(time, nan)
    longitude = full_like(time, nan)
    depth = full_like(time, inf)
    dataset = Dataset(
            data_vars={
                "variance_density": (["time", "frequency"], Szz),
                "a1": (["time", "frequency"], a1),
                "b1": (["time", "frequency"], b1),
                "a2": (["time", "frequency"], a2),
                "b2": (["time", "frequency"], b2),
                "depth": (["time"], depth),
                "latitude": (["time"], latitude),
                "longitude": (["time"], longitude),
            },
            coords={"time": time, "frequency": frequencies},
        )
    return FrequencySpectrum(dataset)


if __name__ == "__main__":
    SPOT_ID = 'SPOT-30808C'
    START_DATE = '2023-02-21'
    END_DATE = '2023-02-24'
    ndbc,wh = get_ndbc()

    full_spec=  load('/Users/pietersmit/Downloads/spectrum.zip')

    # parser = argparse.ArgumentParser(description="power spectrum")
    # parser.add_argument("spccsv", help=("Path to SPC data csv file"))
    # args = parser.parse_args()
    # if not os.path.exists(args.spccsv):
    #     print("Path to csv does not exist!")
    #     exit(1)
    # d = read_spectra(args.spccsv)


    r = requests.get(f'https://kaleidoscope-prod.herokuapp.com/api/data/v1/868050044438386/spectral-data?processing_rate_mins=60&started_at={START_DATE}&ended_at={END_DATE}', headers={"token":"MEFPxIfyKgSVrD6fLITiQvBTG7SbAToEvXOS8Pq3"})

    wind = []
    time = []
    for item in r.json()['data']['results']:

        wind.append(item['results']['wind']['speed'])
        time.append(datetime.strptime(item['end'], "%Y-%m-%dT%H:%M:%S.%fZ"))
    weird_spectra = pullHdrDataIntoRougeWave( r.json()['data']['results'])
    winddf = pd.DataFrame(list(zip(time, wind)),
               columns =['Timestamp (UTC)', 'U(m/s)'])

    spotter_ids = [SPOT_ID]

    # # Time range of interest
    start_date = datetime( int(START_DATE.split('-')[0]),int(START_DATE.split('-')[1]),int(START_DATE.split('-')[2]) )
    end_date = datetime( int(END_DATE.split('-')[0]),int(END_DATE.split('-')[1]),int(END_DATE.split('-')[2]) ) + timedelta(seconds=1)

    # # Get spectra from the API. Under the hood this calls the sofarapi. The returned object is a "spectrum" object which
    # # is a wrapper class around the spectral information (with a ton of convinience functionality)
    spectrum = get_spotter_data(spotter_ids,data_type='frequencyData',start_date=start_date,end_date=end_date)[SPOT_ID]
    spectrum = spectrum.bandpass(fmax=0.5)

    wind = get_spotter_data(spotter_ids,data_type='wind',start_date=start_date,end_date=end_date)

    weird_algo = estimate_u10_from_spectrum(weird_spectra, "peak", fmax=0.8, direction_convention='coming_from_clockwise_north', phillips_constant_beta=0.014, charnock_constant=0.018)
    weird_spectra2 = spotter_frequency_response_correction(weird_spectra)
    weird_algo2 = estimate_u10_from_spectrum(weird_spectra2, "peak", fmax=0.8,
                                            direction_convention='coming_from_clockwise_north',
                                            phillips_constant_beta=0.014, charnock_constant=0.018)

    # Estimate wind speed using the old and new algorithm
    old_algorithm = estimate_u10_from_spectrum(spectrum, "mean")
    new_algorithm = estimate_u10_from_spectrum(spectrum, "peak", fmax=0.8, direction_convention='coming_from_clockwise_north', phillips_constant_beta=0.014, charnock_constant=0.018)

    plt.figure(figsize=[8,6])
    #plt.plot( old_algorithm['time'],old_algorithm['u10'],'k',label='old' )
    plt.plot( new_algorithm['time'],new_algorithm['u10'],'r-+',label='new' )
    #plt.plot( new_algorithm['time'],new_algorithm['u10'],'r',label='new' )
    plt.plot( wind['time'], wind['windVelocity10Meter'], 'g-o', label='embedded' )
    #plt.plot( wind['time'], wind['windVelocity10Meter'], 'g', label='embedded' )
    #plt.plot(winddf['Timestamp (UTC)'], winddf['U(m/s)'], 'bx',label='kalidescope')
    #plt.plot(winddf['Timestamp (UTC)'], winddf['U(m/s)'], 'b',label='kalidescope')
    plt.plot(weird_algo['time'], weird_algo['u10'], 'y-+', label='weird')
    plt.plot(weird_algo['time'], weird_algo2['u10'], 'k', label='weird2')
    #plt.plot(weird_algo['time'], weird_algo['u10'], 'y', label='weird')

    plt.xlim(  [datetime(2023,2,21),datetime(2023,2,24)] )



    plt.grid('on')
    plt.ylabel('U (m/s)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot( spectrum.frequency, spectrum.variance_density.values[-5,:] *spectrum.frequency.values**4 )
    A = weird_spectra.variance_density.values[-5, :]*weird_spectra.frequency.values**4
    plt.plot(weird_spectra.frequency,A.transpose())

    from roguewave.timeseries_analysis.time_integration import complex_response


    B = full_spec.variance_density.values[-3, :]*full_spec.frequency.values**4 / 1000000 / (2.5/256)
    plt.plot(full_spec.frequency,B.transpose(),'k')
    corrected = spotter_frequency_response_correction(full_spec)

    full_spec = corrected
    B = full_spec.variance_density.values[-3, :]*full_spec.frequency.values**4 / 1000000 / (2.5/256)
    plt.plot(full_spec.frequency,B.transpose(),'r')


    #plt.yscale('log')
    plt.grid('on')
    plt.show()

