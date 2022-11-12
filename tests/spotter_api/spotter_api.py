from dotenv import load_dotenv
from pandas.testing import assert_frame_equal
from pandas import DataFrame

try:
    # Load a dotenv file with the eng account if provided, otherwise it is assumed that a proper env is setup
    load_dotenv("/Users/pietersmit/eng.env")
except Exception:
    pass


from roguewave.spotterapi.spotterapi import get_data, get_bulk_wave_data
from datetime import datetime

# Smartmooring unit on the eng account.
SMARTMOORING_TEST_SPOTTER_ID = "SPOT-30083D"
TEST_SPOTTER_ID = "SPOT-010288"
START_DATE = datetime(2022, 11, 10)
END_DATE = datetime(2022, 11, 11)

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
DATAFRAME_KEYS_SST = ["latitude", "longitude", "seaSurfaceTemperature"]


def tst_get_data_waveheight():
    single = get_data(
        TEST_SPOTTER_ID, START_DATE, END_DATE, include_waves=True, cache=False
    )
    assert TEST_SPOTTER_ID in single

    as_list = get_data(
        [TEST_SPOTTER_ID], START_DATE, END_DATE, include_waves=True, cache=False
    )
    assert TEST_SPOTTER_ID in as_list

    data = single[TEST_SPOTTER_ID]
    assert "waves" in data
    assert isinstance(data["waves"], DataFrame)

    directly = get_bulk_wave_data(TEST_SPOTTER_ID, START_DATE, END_DATE, cache=False)[
        TEST_SPOTTER_ID
    ]
    assert_frame_equal(directly, data["waves"])
    for key in DATAFRAME_KEYS:
        assert key in data["waves"], f"{key} not in data"


def tst_get_pressure():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=False,
        include_barometer_data=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single
    data = single[TEST_SPOTTER_ID]

    for key in DATAFRAME_KEYS_BAROMETER:
        assert key in data["barometerData"], f"{key} not in data"


def tst_get_sst():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=False,
        include_surface_temp_data=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single
    data = single[TEST_SPOTTER_ID]
    for key in DATAFRAME_KEYS_SST:
        assert key in data["surfaceTemp"], f"{key} not in data"


def tst_get_wind():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=False,
        include_wind=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single
    data = single[TEST_SPOTTER_ID]
    for key in DATAFRAME_KEYS_WIND:
        assert key in data["wind"], f"{key} not in data"


def tst_get_all():
    single = get_data(
        TEST_SPOTTER_ID,
        START_DATE,
        END_DATE,
        include_waves=True,
        include_wind=True,
        include_surface_temp_data=True,
        include_barometer_data=True,
        include_frequency_data=True,
        cache=False,
    )
    assert TEST_SPOTTER_ID in single
    data = single[TEST_SPOTTER_ID]

    for key in ["wind", "waves", "frequencyData", "barometerData", "surfaceTempData"]:
        assert key in data, f"{key} not in data."


def tst_get_spectrum():
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


if __name__ == "__main__":
    data = tst_get_data_waveheight()
    data = tst_get_pressure()
    data = tst_get_sst()
    data = tst_get_wind()
    data = tst_get_all()
