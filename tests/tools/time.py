from roguewave.tools.time import (
    time_from_timeint,
    date_from_dateint,
    datetime_from_time_and_date_integers,
    to_datetime64,
    to_datetime_utc,
    datetime_to_iso_time_string,
)
from datetime import datetime, timezone, timedelta
from numpy import datetime64, ndarray, dtype
from xarray import DataArray

TEST_DATETIME = datetime(2022, 11, 9, 10, 20, 42, tzinfo=timezone.utc)
UNAWARE_TEST_DATETIME = datetime(2022, 11, 9, 10, 20, 42)
TEST_DATE = datetime(2022, 11, 9, 0, 0, 0, tzinfo=timezone.utc)
TEST_DATE_TIME_INT = (20221109, 102042)
TIMEDELTA = timedelta(seconds=10 * 3600 + 60 * 20 + 42)

XARRAY = DataArray(data=[to_datetime64(TEST_DATETIME)])


def test_time_from_timeint():
    assert time_from_timeint(TEST_DATE_TIME_INT[1]), TIMEDELTA
    assert TEST_DATE == date_from_dateint(TEST_DATE_TIME_INT[0])
    assert TEST_DATETIME == datetime_from_time_and_date_integers(*TEST_DATE_TIME_INT)
    assert datetime64(
        TEST_DATETIME.replace(tzinfo=None), "s"
    ) == datetime_from_time_and_date_integers(*TEST_DATE_TIME_INT, as_datetime64=True)


def test_datetime_conversions():
    # Convert a datetime64, a isoformat, string,string,string, integer,float, datetime
    times = [
        to_datetime64(TEST_DATETIME),
        TEST_DATETIME.isoformat(),
        datetime_to_iso_time_string(TEST_DATETIME),
        "2022-11-09T10:20:42Z",
        "2022-11-09T10:20:42",
        int(TEST_DATETIME.timestamp()),
        float(TEST_DATETIME.timestamp()),
        TEST_DATETIME,
        UNAWARE_TEST_DATETIME,
    ]
    datetimes = to_datetime_utc(times)

    for index, date in enumerate(datetimes):
        assert (
            date == TEST_DATETIME
        ), f"date {date} does not match {TEST_DATETIME} for input {times[index]} at {index}"

    datetimes = to_datetime64(datetimes)
    assert datetimes[0].dtype == dtype("<M8[s]")

    datetimes = to_datetime_utc(datetimes)

    for index, date in enumerate(datetimes):
        assert date == TEST_DATETIME

    datetimes = to_datetime_utc(XARRAY)
    for index, date in enumerate(datetimes):
        assert date == TEST_DATETIME

    datetimes = to_datetime64(XARRAY)
    assert isinstance(datetimes, ndarray)
    datetimes = to_datetime_utc(datetimes)
    for index, date in enumerate(datetimes):
        assert date == TEST_DATETIME

    assert to_datetime_utc(None) is None
    assert to_datetime64(None) is None


if __name__ == "__main__":
    test_time_from_timeint()
    test_datetime_conversions()
