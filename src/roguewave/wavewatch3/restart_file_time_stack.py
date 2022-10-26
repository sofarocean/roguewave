import numpy
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Sequence, Union, Tuple, Dict
from tqdm import tqdm
from xarray import Dataset, concat
from roguewave import FrequencyDirectionSpectrum, TrackSet
from roguewave.interpolate.nd_interp import NdInterpolator
from roguewave.tools.time import to_datetime_utc, to_datetime64
from roguewave.wavewatch3.restart_file import RestartFile, MAXIMUM_NUMBER_OF_WORKERS


class RestartFileTimeStack:
    def __init__(self, restart_files: Sequence[RestartFile], parallel=True):
        self._restart_files = restart_files
        self.grid = restart_files[0].grid
        self.depth = restart_files[0].depth
        self._time = to_datetime_utc([x.time for x in restart_files])
        self.parallel = parallel

        # To make the progress bar work we need to store progress across
        # calls somewhere.
        self._progres = {"position": 0, "total": None, "leave": True}

    @property
    def frequency(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of frequencies
        """
        return self.grid.frequencies

    @property
    def direction(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of directions
        """
        return self.grid.directions

    @property
    def latitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of latitudes.
        """
        return self.grid.latitude

    @property
    def longitude(self) -> numpy.ndarray:
        """
        :return: 1D numpy array of longitudes.
        """
        return self.grid.longitude

    def coordinates(
        self, index: Union[slice, int, numpy.ndarray]
    ) -> Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]]:
        """
        Return the latitude and longitude as a function of the linear index.
        :param index: linear index
        :return:  ( latitude(s), longitude(s)
        """
        ilon = self.grid.longitude_index(index)
        ilat = self.grid.latitude_index(index)
        return self.latitude[ilat], self.longitude[ilon]

    @property
    def number_of_directions(self) -> int:
        """
        :return: number of directions
        """
        return len(self.direction)

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)

    @property
    def number_of_latitudes(self) -> int:
        """
        :return: number of latitudes.
        """
        return len(self.latitude)

    @property
    def number_of_longitudes(self) -> int:
        """
        :return: number of longitudes.
        """
        return len(self.longitude)

    @property
    def number_of_spatial_points(self) -> int:
        """
        :return: Number of spatial points in the restart file. This only counts
            the number of sea points and is *not* equal to
            self.number_of_latitudes * self.number_of_longitudes
            Also referred to as "NSEA" in wavewatch III.
        """
        return self.grid.number_of_spatial_points

    @property
    def number_of_spectral_points(self) -> int:
        """
        :return: number of spectral points.
        """
        return self.number_of_frequencies * self.number_of_directions

    @property
    def time(self) -> Sequence[datetime]:
        return self._time

    def __len__(self):
        return len(self._restart_files)

    def __getitem__(self, nargs) -> FrequencyDirectionSpectrum:
        if len(nargs) == 2:
            time_index, linear_index = nargs
        elif len(nargs) == 3:
            time_index, lat_index, lon_index = nargs
            linear_index = self.grid.index(lat_index, lon_index, valid_only=True)
        else:
            raise ValueError("unexpected number of indices")

        if isinstance(time_index, (int, numpy.integer)):
            time_index = [time_index]

        fancy_index = not isinstance(time_index, slice)
        if fancy_index:
            _input = list(zip(time_index, linear_index))
        else:
            time_index = list(range(*time_index.indices(len(self))))
            _input = [(it, linear_index) for it in time_index]

        def _worker(arg) -> Dataset:
            return self._restart_files[arg[0]][arg[1]].dataset.drop("linear_index")

        self._init_progress_bar(len(_input))
        if self.parallel and len(_input) > 1:
            with ThreadPool(processes=MAXIMUM_NUMBER_OF_WORKERS) as pool:
                data = list(
                    tqdm(
                        pool.imap(_worker, _input),
                        total=self._progres["total"],
                        initial=self._progres["position"],
                        leave=self._progres["leave"],
                    )
                )
        else:
            disable_progress_bar = False
            if len(_input) == 1:
                disable_progress_bar = True

            data = list(
                tqdm(
                    map(_worker, _input),
                    total=self._progres["total"],
                    disable=disable_progress_bar,
                    initial=self._progres["position"],
                    leave=self._progres["leave"],
                )
            )
        self._update_progress_bar(len(_input))

        data = concat(data, dim="time")
        data.coords["time"] = to_datetime64([self.time[it] for it in time_index])
        return FrequencyDirectionSpectrum(data)

    def _init_progress_bar(self, total):
        if self._progres["total"] is None:
            self._progres["total"] = total
            self._progres["position"] = 0
            self._progres["leave"] = True
        else:
            if total + self._progres["position"] == self._progres["total"]:
                self._progres["leave"] = True
            else:
                self._progres["leave"] = False

    def _update_progress_bar(self, number):
        self._progres["position"] += number

        if self._progres["position"] == self._progres["total"]:
            self._progres["total"] = None
            self._progres["position"] = 0

    def interpolate(
        self,
        latitude: Union[numpy.ndarray, float],
        longitude: Union[numpy.ndarray, float],
        time: Union[numpy.ndarray, numpy.datetime64, datetime],
    ) -> FrequencyDirectionSpectrum:
        """
        Extract interpolated spectra at given latitudes and longitudes.
        Input can be either a single latitude and longitude pair, or a
        numpy array of latitudes and longitudes.

        :param latitude: latitudes to get interpolated spectra
        :param longitude: longitudes to get interpolated spectra
        :return: Interpolated spectra. Returned data is of  type float32 and
        has the shape:
                (len(indices),
                number_of_frequencies,
                number_of_directions)
        """
        time = to_datetime64(time)
        points = {
            "time": numpy.atleast_1d(time),
            "latitude": numpy.atleast_1d(latitude),
            "longitude": numpy.atleast_1d(longitude),
        }

        periodic_coordinates = {"longitude": 360}

        def _get_data(indices, idims):
            time_index = indices[0]
            index = self.grid.index(
                latitude_index=indices[1], longitude_index=indices[2]
            )

            output = numpy.zeros(
                (len(index), self.number_of_frequencies, self.number_of_directions)
            )
            mask = index >= 0

            output[mask, :, :] = numpy.squeeze(
                self.__getitem__((time_index[mask], index[mask])).variance_density
            )
            output[~mask, :, :] = numpy.nan
            return output

        data_shape = [
            len(self.time),
            len(self.latitude),
            len(self.longitude),
            self.number_of_frequencies,
            self.number_of_directions,
        ]

        data_coordinates = (
            ("time", to_datetime64(self.time)),
            ("latitude", self.latitude),
            ("longitude", self.longitude),
            ("frequency", self.frequency),
            ("direction", self.direction),
        )

        interpolator = NdInterpolator(
            get_data=_get_data,
            data_coordinates=data_coordinates,
            data_shape=data_shape,
            interp_coord_names=list(points.keys()),
            interp_index_coord_name="time",
            data_periodic_coordinates=periodic_coordinates,
            data_period=None,
            data_discont=None,
        )

        self._init_progress_bar(len(latitude) * 8)
        dataset = interpolator.interpolate(points)

        def _get_depth_data(indices, idims):
            index = self.grid.index(
                latitude_index=indices[0], longitude_index=indices[1]
            )
            output = numpy.zeros(len(index))
            mask = index >= 0
            output[mask] = self.depth[index[mask]]
            output[~mask] = numpy.nan
            return output

        depth_points = {
            "latitude": numpy.atleast_1d(latitude),
            "longitude": numpy.atleast_1d(longitude),
        }
        depth_interpolator = NdInterpolator(
            get_data=_get_depth_data,
            data_coordinates=(
                ("latitude", self.latitude),
                ("longitude", self.longitude),
            ),
            data_shape=[self.number_of_latitudes, self.number_of_longitudes],
            interp_coord_names=list(depth_points.keys()),
            interp_index_coord_name="latitude",
            data_periodic_coordinates=periodic_coordinates,
            data_period=None,
            data_discont=None,
        )

        return FrequencyDirectionSpectrum(
            Dataset(
                data_vars={
                    "variance_density": (("time", "frequency", "direction"), dataset),
                    "depth": (("time",), depth_interpolator.interpolate(depth_points)),
                    "longitude": (("time"), longitude),
                    "latitude": (("time"), latitude),
                },
                coords={
                    "time": time,
                    "frequency": self.frequency,
                    "direction": self.direction,
                },
            )
        )

    def interpolate_tracks(
        self, tracks: TrackSet
    ) -> Dict[str, FrequencyDirectionSpectrum]:
        return {
            _id: self.interpolate(x.latitude, x.longitude, x.time)
            for _id, x in tracks.tracks.items()
        }
