from typing import Mapping, Sequence, Union, Tuple
from pandas import DataFrame, concat
from xarray import DataArray
from datetime import timedelta

from roguewave.spotterapi.spotterapi import get_spotter_data
from roguewave.interpolate.dataset import tracks_as_dataset
from roguewave.modeldata.extract import extract_from_remote_dataset
from roguewave.interpolate.dataframe import interpolate_dataframe_time
from roguewave.modeldata.timebase import TimeSlice


# =============================================================================
def colocate_model_spotter(
    variable: Union[Sequence[str], str],
    spotter_ids: Sequence[str],
    time_slice: TimeSlice,
    model_name: str,
    cache_name: str = None,
    parallel: bool = True,
    timebase: str = "model",
    slice_remotely=False,
    return_as_dataset=False,
) -> Union[
    Tuple[Mapping[str, DataFrame], Mapping[str, DataFrame]], Tuple[DataArray, DataArray]
]:
    """
    Colocate spoter output and modek data
    :param variable: name of the variable of interest. Can be a list in which
            case all variables in the list are retrieved.
    :param spotter_ids: List of spotter ids of interest
    :param time_slice: time slice of interest.
    :param model_name: model name
    :param slice_remotely: (default False) if True Skip local cache and try to
    read directly from
        remote. This tries to avoid downloading full files and instead tries
        to only grab the data it needs. Typically slower.
        Does not work for grib files.
    :param parallel: If slice_remotely=True controls whether we download in
        parallel. To note- if parallel is enabled we need to make sure the
        main script is guarded by if __name__ == '__main__'.
    :param cache_name: name of local cache. If None, default cache setup will
        be used.
    :param timebase: Can be ['native','spotter','model'] if
        - native: output for spotters and model at spotters will be generated
                  at their own valid times
        - spotter: model data will be interpolated to spotter reporting time
        - model: Spotter data will be interpolated to model time
    :param return_as_dataset: Instead of a dictornary return everything as one
        xarray dataset ("Pandas on steroids"), where spotter_id is one of the
        dimensions of the datastructure (a "column" if you will)
    :return:
    """

    if return_as_dataset:
        if (not timebase == "model") and (len(spotter_ids) > 1):
            raise ValueError(
                "Cannot return as an xarray dataset if the "
                "time base is native or spotter"
            )

    if timebase.lower() == "model":
        # If we have a model time base we interpolate spotter observations onto the model grid
        # To ensure we have data to use for interpolation at the end points we widen the interval there
        obs_delta = timedelta(hours=3)
    else:
        obs_delta = timedelta(hours=0)

    df = get_spotter_data(
        spotter_ids,
        "waves",
        time_slice.start_time - obs_delta,
        time_slice.end_time + obs_delta,
    )
    spotters = {
        key: value.drop(columns="spotter_id")
        for (key, value) in df.groupby("spotter_id")
    }

    for spotter_id in list(spotters.keys()):
        # If only one value - pop because we cannot interpolate on 1 value.
        if spotters[spotter_id].shape[0] <= 2:
            spotters.pop(spotter_id)

    # we want model data to cover the entire period. Spotter data will be returned up to the enddate. Hence we enforce
    # inclusion of the endpoint during colocation, this will ensure that whenever there is spotter data, there is
    # enclosing model data (which I would argue is the expected behaviour).
    if timebase.lower() == "spotter":
        time_slice.endpoint = True

    model = extract_from_remote_dataset(
        spotters,
        variable,
        time_slice,
        model_name,
        slice_remotely=slice_remotely,
        parallel=parallel,
        cache_name=cache_name,
    )

    for spotter_id in spotters:
        s = spotters[spotter_id]  # type: DataFrame
        m = model[spotter_id]  # type: DataFrame
        model_time = m["time"].values
        if timebase.lower() == "native":
            pass
        elif timebase.lower() == "observed" or timebase.lower() == "spotter":
            m = interpolate_dataframe_time(m, s["time"].values)
        elif timebase.lower() == "model":
            s = interpolate_dataframe_time(s, m["time"].values)
        else:
            raise ValueError(
                f"Unknown timebase {timebase}, must be one of: "
                f"native, observed, model, or spotter"
            )

        model[spotter_id] = m
        spotters[spotter_id] = s

    if return_as_dataset:
        return tracks_as_dataset(model_time, model), tracks_as_dataset(
            model_time, spotters
        )
    else:
        model = colocate_into_single_dataframe(model)
        spotters = colocate_into_single_dataframe(spotters)
        return model, spotters


def colocated_tracks_as_dataset(*args) -> Tuple[DataArray, DataArray]:
    if len(args) == 1:
        """
        for convinience if passing immediately from the track interpolation
        """
        model, spotter = args[0]
    else:
        model, spotter = args

    keys = list(model.keys())
    time = model[keys[0]].index.values
    model = tracks_as_dataset(time, model)
    spotter = tracks_as_dataset(time, spotter)
    return spotter, model


def colocate_into_single_dataframe(data: dict[str, DataFrame]) -> DataFrame:
    _ids = list(data.keys())
    values = [data[_id] for _id in _ids]

    data = concat(values, keys=_ids, names=["spotter_id", "time index"])
    data.reset_index(inplace=True)
    data.drop(columns="time index", inplace=True)
    return data
