"""Object tracking in the BEV world model: one CTRV Kalman filter per track (fsd.ekf).

Two trackers share the filtering and differ only in how a detection finds its track:

* ``GtVelocityTracker`` - by the stable nuScenes ``instance_token`` (free, zero
  association error). The oracle.
* ``PredictionVelocityTracker`` - greedy nearest-by-class association against each
  track's predicted position, for detector boxes (CenterPoint, NMP) that carry no
  identity.

Every update predicts all tracks to the frame's time, updates the matched ones,
lets unmatched ones coast and retires those unseen for longer than ``max_age_s``.
Both trackers reset themselves when the frame's scene changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from fsd.data import SurroundFrame
from fsd.ego_motion import wrap_angle
from fsd.ekf import CtrvEkf
from fsd.object_detection import Box3D


@dataclass(frozen=True)
class TrackedObject:
    box: Box3D                 # the detection as given
    track_id: str
    velocity_ego: np.ndarray   # (vx, vy) m/s, filtered, current ego frame
    speed_mps: float
    future_xy_ego: np.ndarray  # (K, 2) CTRV-predicted centres, current ego frame; empty below min speed
    smoothed: Box3D            # the detection moved to the filtered centre and heading


@dataclass
class _Track:
    class_name: str
    ekf: CtrvEkf
    seen_us: int


class _EkfTracker:
    def __init__(self, horizons_s: tuple[float, ...], min_speed_mps: float, max_age_s: float):
        self.horizons_s = horizons_s
        self.min_speed_mps = min_speed_mps
        self.max_age_s = max_age_s
        self._scene_token: str | None = None
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[str, _Track] = {}
        self._now: int | None = None

    def _associate(self, boxes: list[Box3D]) -> dict[int, str]:
        """Detection index -> existing track id."""
        raise NotImplementedError

    def _new_id(self, box: Box3D) -> str:
        raise NotImplementedError

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        """One TrackedObject per detection, in order."""
        if frame.scene_token != self._scene_token:
            self.reset()
            self._scene_token = frame.scene_token
        if self._now is not None and frame.timestamp_us > self._now:
            for track in self._tracks.values():
                track.ekf.predict((frame.timestamp_us - self._now) / 1_000_000.0)
        self._now = frame.timestamp_us
        match = self._associate(boxes, frame) if self._tracks else {}
        out = []
        for i, box in enumerate(boxes):
            z = _measurement(box, frame)
            track_id = match.get(i)
            if track_id is None:
                track_id = self._new_id(box)
                self._tracks[track_id] = _Track(box.class_name, CtrvEkf(*z), frame.timestamp_us)
            else:
                track = self._tracks[track_id]
                # A box looks the same from either end: read a reversed heading as the track's own.
                if abs(wrap_angle(z[2] - track.ekf.x[3])) > math.pi / 2:
                    z[2] = wrap_angle(z[2] + math.pi)
                track.ekf.update(z)
                track.seen_us = frame.timestamp_us
            out.append(self._readout(track_id, box, frame))
        self._tracks = {k: t for k, t in self._tracks.items() if (frame.timestamp_us - t.seen_us) / 1_000_000.0 <= self.max_age_s}
        return out

    def _readout(self, track_id: str, box: Box3D, frame: SurroundFrame) -> TrackedObject:
        ekf, pose = self._tracks[track_id].ekf, frame.ego_pose
        px, py, v, yaw, _ = ekf.x
        z = pose.apply(box.center_ego[None])[0, 2]
        velocity = pose.rotation[:2, :2].T @ (v * np.array([math.cos(yaw), math.sin(yaw)]))
        future = np.empty((0, 2))
        if abs(v) >= self.min_speed_mps:
            ahead = ekf.forecast(self.horizons_s)
            future = pose.invert(np.c_[ahead, np.full(len(ahead), z)])[:, :2]
        centre = pose.invert(np.array([[px, py, z]]))[0]
        smoothed = replace(box, center_ego=centre, yaw_ego=float(wrap_angle(yaw - pose.yaw)))
        return TrackedObject(box, track_id, velocity, float(abs(v)), future, smoothed)


def _measurement(box: Box3D, frame: SurroundFrame) -> np.ndarray:
    """(px, py, yaw) of a detection in the global frame."""
    xy = frame.ego_pose.apply(box.center_ego[None])[0, :2]
    return np.array([xy[0], xy[1], wrap_angle(frame.ego_pose.yaw + box.yaw_ego)])


class GtVelocityTracker(_EkfTracker):
    """Tracks keyed by the stable nuScenes instance token (the oracle)."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5, max_age_s: float = 0.5):
        super().__init__(horizons_s, min_speed_mps, max_age_s)

    def _associate(self, boxes: list[Box3D], frame: SurroundFrame) -> dict[int, str]:
        return {i: box.instance_token for i, box in enumerate(boxes) if box.instance_token in self._tracks}

    def _new_id(self, box: Box3D) -> str:
        return box.instance_token


class PredictionVelocityTracker(_EkfTracker):
    """Greedy nearest-by-class association against predicted track positions, for identity-free detections."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5, max_age_s: float = 0.5, gate_m: float = 5.0):
        super().__init__(horizons_s, min_speed_mps, max_age_s)
        self.gate_m = gate_m
        self._next_id = 0

    def _new_id(self, box: Box3D) -> str:
        self._next_id += 1
        return f"p{self._next_id - 1}"

    def _associate(self, boxes: list[Box3D], frame: SurroundFrame) -> dict[int, str]:
        if not boxes:
            return {}
        ids = list(self._tracks)
        predicted = np.array([self._tracks[k].ekf.x[:2] for k in ids])
        xy = np.array([_measurement(box, frame)[:2] for box in boxes])
        dist = np.linalg.norm(xy[:, None] - predicted[None], axis=2)
        dist[np.array([b.class_name for b in boxes])[:, None] != np.array([self._tracks[k].class_name for k in ids])[None]] = np.inf
        pairs = np.argwhere(dist <= self.gate_m)
        match: dict[int, str] = {}
        for i, j in pairs[np.argsort(dist[pairs[:, 0], pairs[:, 1]], kind="stable")]:
            if i not in match and ids[j] not in match.values():
                match[int(i)] = ids[j]
        return match
