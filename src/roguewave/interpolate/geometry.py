import numpy
from dataclasses import dataclass
from typing import Sequence, Mapping, List, Union
from pandas import DataFrame
from roguewave.tools.time import to_datetime64, to_datetime_utc
from datetime import datetime
from roguewave.interpolate.general import interpolate_periodic
from numbers import Number


class _Geometry:
    pass


@dataclass()
class SpatialPoint(_Geometry):
    latitude: float
    longitude: float
    id: str

    @property
    def is_valid(self) -> bool:
        return numpy.isfinite(self.latitude) and numpy.isfinite(self.longitude)


@dataclass()
class SpaceTimePoint(SpatialPoint):
    time: datetime

    @classmethod
    def from_spatial_point(cls, point: SpatialPoint, time: datetime):
        return cls(point.latitude, point.longitude, point.id, time)


@dataclass()
class Cluster(_Geometry):
    """
    A cluster is a set of points, each identified by unique id.
    """

    points: Mapping[str, SpatialPoint]

    @classmethod
    def from_lat_lon_arrays(cls, lats, lons):
        points = {}
        for index in range(0, len(lats)):
            _id = str(index)
            points[_id] = SpatialPoint(lats[index], lons[index], _id)

        return cls(points)

    @property
    def latitude(self) -> numpy.ndarray:
        return numpy.array([x.latitude for x in self.points.values()])

    @property
    def longitude(self) -> numpy.ndarray:
        return numpy.array([x.longitude for x in self.points.values()])

    @property
    def ids(self) -> Sequence[str]:
        return [x for x in self.points.keys()]


@dataclass()
class ClusterStack(_Geometry):
    """
    A cluster timestack is a stack of clusters in time, e.g. a cluster of
    spotters as it evolves in time.
    """

    time: numpy.ndarray
    clusters: Sequence[Cluster]

    def as_track_set(self) -> "TrackSet":
        return TrackSet.from_clusters(self)

    @classmethod
    def from_track_set(cls, track_set: "TrackSet", time):
        return track_set.as_cluster_time_stack(time)

    def __len__(self):
        return len(self.time)


class Track(_Geometry):
    """
    A track is  the drift track of a single buoy in time
    """

    def __init__(self, points: List[SpaceTimePoint], id):
        points = list(filter(lambda x: x.is_valid, points))
        self.points = points
        self.id = id

    @property
    def time(self) -> numpy.ndarray:
        return to_datetime64([x.time for x in self.points])

    @property
    def latitude(self) -> numpy.ndarray:
        return numpy.array([x.latitude for x in self.points])

    @property
    def longitude(self) -> numpy.ndarray:
        return numpy.array([x.longitude for x in self.points])

    def interpolate(self, target_time) -> "Track":
        target_time = to_datetime64(target_time)
        time = self.time
        latitude = interpolate_periodic(
            time.astype("float64"),
            self.latitude,
            target_time.astype("float64"),
            left=self.latitude[0],
            right=self.latitude[-1],
        )
        longitude = interpolate_periodic(
            time.astype("float64"),
            self.longitude,
            target_time.astype("float64"),
            left=self.longitude[0],
            right=self.longitude[-1],
            fp_period=360,
        )
        return self.from_arrays(latitude, longitude, target_time, self.id)

    @classmethod
    def from_arrays(cls, latitude, longitude, time, id) -> "Track":
        points = []
        for lat, lon, t in zip(latitude, longitude, time):
            points.append(SpaceTimePoint(lat, lon, id, to_datetime_utc(t)))
        return cls(points, id)

    @classmethod
    def from_spotter(cls, spotter_id, spotter):
        points = []
        if isinstance(spotter, DataFrame):
            for latitude, longitude, time in zip(
                spotter["latitude"].values,
                spotter["longitude"].values,
                spotter["time"].values,
            ):
                points.append(
                    SpaceTimePoint(
                        time=time, latitude=latitude, longitude=longitude, id=spotter_id
                    )
                )
        elif hasattr(spotter, "dataset"):
            for latitude, longitude, time in zip(
                spotter.dataset["latitude"].values,
                spotter.dataset["longitude"].values,
                spotter.dataset["time"].values,
            ):
                points.append(
                    SpaceTimePoint(
                        time=time, latitude=latitude, longitude=longitude, id=spotter_id
                    )
                )
        else:
            for x in spotter:
                points.append(
                    SpaceTimePoint(
                        time=x.time,
                        latitude=x.latitude,
                        longitude=x.longitude,
                        id=spotter_id,
                    )
                )
        return cls(points, spotter_id)

    def __len__(self):
        return len(self.points)


@dataclass()
class TrackSet(_Geometry):
    """
    A collection of tracks is a set of tracks for multiple buoys.
    """

    tracks: Mapping[str, Track]

    @classmethod
    def from_spotters(cls, spotters: Mapping):
        tracks = {}
        for spotter_id, spotter in spotters.items():
            tracks[spotter_id] = Track.from_spotter(spotter_id, spotter)
        return cls(tracks)

    @classmethod
    def from_clusters(cls, cluster_time_stack: "ClusterStack"):
        tracks = {}
        for cluster, time in zip(cluster_time_stack.clusters, cluster_time_stack.time):

            for _id, point in cluster.points:
                space_time_point = SpaceTimePoint(
                    longitude=point.latitude,
                    latitude=point.longitude,
                    time=time,
                    id=_id,
                )
                if _id not in tracks:
                    tracks[_id] = Track([space_time_point], _id)
                else:
                    tracks[_id].points.append(space_time_point)
        return cls(tracks)

    def as_cluster_time_stack(self, time):
        tracks = self.interpolate(time)

        clusters = [{} for x in range(len(time))]
        for _id, track in tracks.tracks.items():
            for ii, point in enumerate(track.points):
                if point.is_valid:
                    index = numpy.searchsorted(
                        time, [to_datetime64(point.time)], side="left"
                    )[0]

                    clusters[index][_id] = SpatialPoint(
                        point.latitude, point.longitude, _id
                    )

        clusters = [Cluster(x) for x in clusters]
        return ClusterStack(time, clusters)

    def interpolate(self, time) -> "TrackSet":
        time = to_datetime64(time)
        tracks = {}
        for id, track in self.tracks.items():
            tracks[id] = track.interpolate(time)
        return TrackSet(tracks)

    @classmethod
    def from_cluster(cls, cluster: Cluster, time: numpy.ndarray) -> "TrackSet":
        tracks = {}
        for _id, point in cluster.points.items():
            track_points = [SpaceTimePoint.from_spatial_point(point, t) for t in time]
            tracks[point.id] = Track(points=track_points, id=point.id)
        return TrackSet(tracks)


Geometry = Union[
    _Geometry, Sequence[numpy.ndarray], Sequence[Sequence], Sequence[Number], Mapping
]


def convert_to_cluster_stack(geometry: Geometry, time: numpy.ndarray) -> ClusterStack:

    if isinstance(geometry, TrackSet):
        return geometry.as_cluster_time_stack(time)

    elif isinstance(geometry, Cluster):
        return ClusterStack(time, [geometry for _ in time])

    elif isinstance(geometry, Track):
        tracks = TrackSet({"track": geometry})
        return ClusterStack.from_track_set(tracks, time)

    elif isinstance(geometry, ClusterStack):
        return geometry

    elif isinstance(geometry, Sequence):
        tracks = convert_to_track_set(geometry, time)
        return ClusterStack.from_track_set(tracks, time)

    elif isinstance(geometry, Mapping):
        tracks = convert_to_track_set(geometry, time)
        return ClusterStack.from_track_set(tracks, time)


def convert_to_track_set(geometry: Geometry, time: numpy.ndarray) -> TrackSet:

    if isinstance(geometry, TrackSet):
        return geometry.interpolate(time)

    elif isinstance(geometry, Cluster):
        return TrackSet.from_clusters(ClusterStack(time, [geometry for _ in time]))

    elif isinstance(geometry, Track):
        return TrackSet({"track": geometry.interpolate(time)})

    elif isinstance(geometry, ClusterStack):
        return geometry.as_track_set()

    elif isinstance(geometry, Sequence):
        if not (isinstance(geometry[0], Sequence) or isinstance(geometry[0], Sequence)):
            _geometry = [[geometry[0]], [geometry[1]]]
        else:
            _geometry = geometry

        cluster = Cluster.from_lat_lon_arrays(_geometry[0], _geometry[1])
        return TrackSet.from_cluster(cluster, time)

    elif isinstance(geometry, Mapping):
        return TrackSet.from_spotters(geometry).interpolate(time)
