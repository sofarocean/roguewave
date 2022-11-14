from roguewave import colocate_model_spotter, TimeSliceForecast, to_datetime_utc
from datetime import datetime, timezone, timedelta
from numpy.testing import assert_array_equal, assert_allclose
from numpy import squeeze

TEST_SPOTTER_ID = "SPOT-010288"
START_DATE = datetime(2022, 11, 10, tzinfo=timezone.utc)
END_DATE = datetime(2022, 11, 11, tzinfo=timezone.utc)
MODEL = "SofarECMWFHResOperationalWaveModel0p25"


def tst_colocate_waveheight_timebase_model():
    variable = "significantWaveHeight"
    time_slice = TimeSliceForecast(START_DATE, END_DATE - START_DATE)

    model, spotter = colocate_model_spotter(
        variable,
        [TEST_SPOTTER_ID],
        time_slice,
        MODEL,
        timebase="model",
        return_as_dataset=True,
    )

    sig_mod = model.sel({"variables": variable})
    sig_obs = spotter.sel({"variables": variable})

    assert to_datetime_utc(sig_mod["time"])[0] == START_DATE
    assert to_datetime_utc(sig_mod["time"])[-1] == END_DATE
    assert to_datetime_utc(sig_obs["time"])[0] == START_DATE
    assert to_datetime_utc(sig_obs["time"])[-1] == END_DATE

    model, spotter = colocate_model_spotter(
        variable,
        [TEST_SPOTTER_ID],
        time_slice,
        MODEL,
        timebase="model",
        return_as_dataset=False,
    )
    assert_array_equal(model[variable].values, squeeze(sig_mod.values))
    assert_array_equal(spotter[variable].values, squeeze(sig_obs.values))

    assert to_datetime_utc(model["time"])[0] == START_DATE
    assert to_datetime_utc(model["time"])[-1] == END_DATE
    assert to_datetime_utc(spotter["time"])[0] == START_DATE
    assert to_datetime_utc(spotter["time"])[-1] == END_DATE

    # Canary values to see if anything changes.
    indices = [1, 5, 7]
    values_obs = [5.58229306, 6.0206125, 6.28336806]
    values_mod = [5.31125933, 5.82403512, 6.08862596]

    ii = -1
    for index in indices:
        ii += 1
        assert_allclose(model[variable].values[index], values_mod[ii])
        assert_allclose(spotter[variable].values[index], values_obs[ii])


def tst_colocate_waveheight_timebase_spotter():
    variable = "significantWaveHeight"
    time_slice = TimeSliceForecast(START_DATE, END_DATE - START_DATE)

    model, spotter = colocate_model_spotter(
        variable,
        [TEST_SPOTTER_ID],
        time_slice,
        MODEL,
        timebase="spotter",
        return_as_dataset=False,
    )

    assert to_datetime_utc(model["time"])[0] == to_datetime_utc(spotter["time"])[0]
    assert to_datetime_utc(model["time"])[-1] == to_datetime_utc(spotter["time"])[-1]
    assert to_datetime_utc(spotter["time"])[0] - START_DATE < timedelta(hours=1)
    assert END_DATE - to_datetime_utc(spotter["time"])[-1] < timedelta(hours=1)

    # Canary values to see if anything changes.
    indices = [1, 5, 7]
    values_obs = [5.625, 5.874, 6.334]
    values_mod = [5.3525547, 5.87997223, 6.12686167]

    ii = -1
    for index in indices:
        ii += 1
        assert_allclose(model[variable].values[index], values_mod[ii])
        assert_allclose(spotter[variable].values[index], values_obs[ii])


if __name__ == "__main__":
    tst_colocate_waveheight_timebase_model()
    tst_colocate_waveheight_timebase_spotter()
