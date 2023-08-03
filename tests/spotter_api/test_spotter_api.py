from dotenv import load_dotenv
from pandas.testing import assert_frame_equal
from pandas import DataFrame
from roguewave.spotterapi import spotterapi

try:
    # Load a dotenv file with the eng account if provided, otherwise it is assumed that a proper env is setup
    load_dotenv("/Users/pietersmit/eng.env")
except Exception:
    pass

from roguewave.spotterapi.spotterapi import (
    get_data,
    get_bulk_wave_data,
    get_spotter_data,
    get_spectrum,
)
from datetime import datetime, timedelta
from roguewave import to_datetime_utc, FrequencySpectrum

# Smartmooring unit on the eng account.
SMARTMOORING_TEST_SPOTTER_ID = "SPOT-30083D"
TEST_SPOTTER_ID = "SPOT-010288"
START_DATE_MULTIPAGE = datetime(2022, 10, 10)
START_DATE = datetime(2022, 11, 10)
END_DATE = datetime(2022, 11, 11)

START_DATE_SM = datetime(2022, 9, 10)
END_DATE_SM = datetime(2022, 9, 21)

DATAFRAME_KEYS = [
    "latitude",
    "longitude",
    "significantWaveHeight",
    "peakPeriod",
    "meanPeriod",
    "peakDirection",
    "peakDirectionalSpread",
    "meanDirection",
    "meanDirectionalSpread",
    "peakFrequency",
]

DATAFRAME_KEYS_BAROMETER = [
    "latitude",
    "longitude",
    "units",
    "unit_type",
    "data_type_name",
    "seaSurfacePressure",
]

DATAFRAME_KEYS_WIND = [
    "latitude",
    "longitude",
    "windVelocity10Meter",
    "windDirection10Meter",
    "seasurfaceId",
]

DATAFRAME_KEYS_SMARTMOORING = [
    "latitude",
    "longitude",
    "time",
    "sensorPosition",
    "units",
    "value",
    "unit_type",
    "data_type_name",
]

DATAFRAME_KEYS_SST = ["latitude", "longitude", "seaSurfaceTemperature"]

DATAFRAME_KEYS_MICROPHONEDATA = [
    "latitude",
    "longitude",
    "units",
    "soundPressure",
    "unit_type",
    "data_type_name",
]


def test_get_data_waveheight():
    single = get_spotter_data(
        TEST_SPOTTER_ID,'waves', START_DATE, END_DATE,  cache=False
    )
    assert isinstance(single, DataFrame)
    assert single['spotter_id'][1] == TEST_SPOTTER_ID

    as_list = get_spotter_data(
        [TEST_SPOTTER_ID], 'waves',START_DATE, END_DATE, cache=False
    )
    assert isinstance(as_list, DataFrame)

    directly = get_bulk_wave_data(TEST_SPOTTER_ID, START_DATE, END_DATE, cache=False)

    assert_frame_equal(directly[TEST_SPOTTER_ID], as_list)
    for key in DATAFRAME_KEYS:
        assert key in as_list, f"{key} not in data"

    single = get_spotter_data(
        TEST_SPOTTER_ID,
        "waves",
        START_DATE,
        END_DATE,
        cache=False,
    )
    for key in DATAFRAME_KEYS:
        assert key in single, f"{key} not in data"


def test_get_pressure():
    single = get_spotter_data(
        TEST_SPOTTER_ID,
        'barometerData',
        START_DATE,
        END_DATE,
        cache=False,
    )
    assert isinstance(single, DataFrame)
    assert single['spotter_id'][1] == TEST_SPOTTER_ID

    for key in DATAFRAME_KEYS_BAROMETER:
        assert key in single, f"{key} not in data"


def test_get_sst():

    single = get_spotter_data(
        TEST_SPOTTER_ID,
        "surfaceTemp",
        START_DATE,
        END_DATE,
        cache=False,
    )
    for key in DATAFRAME_KEYS_SST:
        assert key in single, f"{key} not in data"


def test_get_wind():
    single = get_spotter_data(
        TEST_SPOTTER_ID,
        "wind",
        START_DATE,
        END_DATE,
        cache=False,
    )
    for key in DATAFRAME_KEYS_WIND:
        assert key in single, f"{key} not in data"


def test_get_smart_mooring_data():
    single = get_spotter_data(
        SMARTMOORING_TEST_SPOTTER_ID,
        "smartMooringData",
        START_DATE_SM,
        END_DATE_SM,
        cache=False,
    )

    for key in DATAFRAME_KEYS_SMARTMOORING:
        assert key in single, f"{key} not in data"

    # Make sure our artificial rate limiting does not influence results returned.
    spotterapi.MAX_DAYS_SMARTMOORING = 20
    no_rate_limit = get_spotter_data(
        SMARTMOORING_TEST_SPOTTER_ID,
        "smartMooringData",
        START_DATE_SM,
        END_DATE_SM,
        cache=False,
    )

    assert_frame_equal(single, no_rate_limit)


def test_get_microphone():
    single = get_spotter_data(
        TEST_SPOTTER_ID,
        "microphoneData",
        START_DATE,
        END_DATE,
        cache=False,
    )

    data = single
    for key in DATAFRAME_KEYS_MICROPHONEDATA:
        assert key in data, f"{key} not in data"


def test_get_all():
    pass


def test_get_spectrum():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=False,
        include_wind=False,
        include_surface_temp_data=False,
        include_barometer_data=False,
        include_frequency_data=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single

    single = get_spotter_data(
        TEST_SPOTTER_ID,
        "frequencyData",
        START_DATE,
        END_DATE,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single

    single = get_spectrum(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single


def test_get_paged_data():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=False,
        include_wind=False,
        include_surface_temp_data=False,
        include_barometer_data=False,
        include_frequency_data=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single


def test_multi_page():
    data = get_spotter_data(
        [TEST_SPOTTER_ID, SMARTMOORING_TEST_SPOTTER_ID],
        "waves",  # ,'frequencyData','surfaceTemp'
        START_DATE_MULTIPAGE,
        END_DATE,
        cache=False,
    )

    data = data[data["spotter_id"] == TEST_SPOTTER_ID]
    assert isinstance(data, DataFrame)
    assert data.shape == (766, 13)
    assert data["significantWaveHeight"][600] == 6.73
    assert to_datetime_utc(END_DATE) - to_datetime_utc(data["time"].iloc[-1]) < timedelta(hours=1)
    assert to_datetime_utc(data["time"].iloc[0]) - to_datetime_utc(START_DATE) < timedelta(hours=1)

    data = get_spotter_data(
        [TEST_SPOTTER_ID, SMARTMOORING_TEST_SPOTTER_ID],
        "surfaceTemp",  # ,'frequencyData','surfaceTemp'
        START_DATE_MULTIPAGE,
        END_DATE,
        cache=False,
    )

    data = data[data["spotter_id"] == TEST_SPOTTER_ID]
    assert isinstance(data, DataFrame)
    assert data.shape == (2298, 6)
    assert to_datetime_utc(END_DATE) - to_datetime_utc(data["time"].iloc[-1]) < timedelta(hours=1)
    assert to_datetime_utc(data["time"].iloc[0]) - to_datetime_utc(START_DATE) < timedelta(hours=1)

    data = get_spotter_data(
        [TEST_SPOTTER_ID, SMARTMOORING_TEST_SPOTTER_ID],
        "frequencyData",  # ,'frequencyData','surfaceTemp'
        START_DATE_MULTIPAGE,
        END_DATE,
        cache=False,
    )
    data = data[TEST_SPOTTER_ID]
    assert isinstance(data, FrequencySpectrum)
    assert len(data) == 766
    assert abs(data.significant_waveheight[600] - 6.73) < 0.01
    assert to_datetime_utc(END_DATE) - to_datetime_utc(data.time)[-1] < timedelta(
        hours=1
    )
    assert to_datetime_utc(data.time)[0] - to_datetime_utc(START_DATE) < timedelta(
        hours=1
    )


if __name__ == "__main__":
    test_get_smart_mooring_data()
    test_multi_page()
    test_get_data_waveheight()
    test_get_pressure()
    test_get_sst()
    test_get_wind()
    test_get_all()
    test_get_microphone()
