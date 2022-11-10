from roguewave.tools.time import (
    time_from_timeint,
    date_from_dateint,
    datetime_from_time_and_date_integers,
)
from datetime import datetime, timezone, timedelta
from numpy import datetime64

TEST_DATETIME = datetime(2022, 11, 9, 10, 20, 42, tzinfo=timezone.utc)
TEST_DATE = datetime(2022, 11, 9, 0, 0, 0, tzinfo=timezone.utc)
TEST_DATE_TIME_INT = (20221109, 102042)
TIMEDELTA = timedelta(seconds=10 * 3600 + 60 * 20 + 42)


def t_time_from_timeint():
    assert time_from_timeint(TEST_DATE_TIME_INT[1]), TIMEDELTA
    assert TEST_DATE == date_from_dateint(TEST_DATE_TIME_INT[0])
    assert TEST_DATETIME == datetime_from_time_and_date_integers(*TEST_DATE_TIME_INT)
    print(datetime_from_time_and_date_integers(*TEST_DATE_TIME_INT, as_datetime64=True))
    assert datetime64(
        TEST_DATETIME.replace(tzinfo=None), "s"
    ) == datetime_from_time_and_date_integers(*TEST_DATE_TIME_INT, as_datetime64=True)


if __name__ == "__main__":
    t_time_from_timeint()
