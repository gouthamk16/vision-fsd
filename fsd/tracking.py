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


# Coast through up to two missed 2 Hz keyframes. Keyframes are only ~0.5 s apart, so a limit of exactly one period
# would retire a track on one miss or not depending on timestamp jitter.
MAX_AGE_S = 1.2


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
        self._count = 0

    def _associate(self, boxes: list[Box3D], measured: np.ndarray) -> dict[int, str]:
        """Detection index -> existing track id; `measured` is each detection's global (px, py, yaw)."""
        raise NotImplementedError

    def _new_id(self, box: Box3D) -> str:
        raise NotImplementedError

    def _fresh_id(self, prefix: str) -> str:
        self._count += 1
        return f"{prefix}{self._count - 1}"

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        """One TrackedObject per detection, in order."""
        # A new scene, or a frame no later than the last (repeated or rewound), starts over: updating without a
        # predict step would make the filters overconfident and age tracks backwards.
        if frame.scene_token != self._scene_token or (self._now is not None and frame.timestamp_us <= self._now):
            self.reset()
            self._scene_token = frame.scene_token
        if self._now is not None:
            for track in self._tracks.values():
                track.ekf.predict((frame.timestamp_us - self._now) / 1_000_000.0)
        self._now = frame.timestamp_us
        measured = np.array([_measurement(box, frame) for box in boxes]).reshape(-1, 3)
        match = self._associate(boxes, measured) if self._tracks else {}
        out = []
        for i, box in enumerate(boxes):
            z = measured[i].copy()
            track_id = match.get(i)
            if track_id is None:
                track_id = self._new_id(box)
                self._tracks[track_id] = _Track(box.class_name, CtrvEkf(*z), frame.timestamp_us)
            else:
                track = self._tracks[track_id]
                # A box looks the same from either end: read a reversed heading (~180 deg off) as the track's own.
                # Real turns stay well under 120 deg from the predicted heading, even after coasting 1.2 s.
                if abs(wrap_angle(z[2] - track.ekf.x[3])) > math.radians(120.0):
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

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5, max_age_s: float = MAX_AGE_S):
        super().__init__(horizons_s, min_speed_mps, max_age_s)

    def _associate(self, boxes: list[Box3D], measured: np.ndarray) -> dict[int, str]:
        return {i: box.instance_token for i, box in enumerate(boxes) if box.instance_token and box.instance_token in self._tracks}

    def _new_id(self, box: Box3D) -> str:
        # A box without a token can't be followed: a track of its own that nothing will match again.
        return box.instance_token or self._fresh_id("untracked")


class PredictionVelocityTracker(_EkfTracker):
    """Greedy nearest-by-class association against predicted track positions, for identity-free detections."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5, max_age_s: float = MAX_AGE_S, gate_m: float = 5.0):
        super().__init__(horizons_s, min_speed_mps, max_age_s)
        self.gate_m = gate_m

    def _new_id(self, box: Box3D) -> str:
        return self._fresh_id("p")

    def _associate(self, boxes: list[Box3D], measured: np.ndarray) -> dict[int, str]:
        if not boxes:
            return {}
        ids = list(self._tracks)
        predicted = np.array([self._tracks[k].ekf.x[:2] for k in ids])
        dist = np.linalg.norm(measured[:, None, :2] - predicted[None], axis=2)
        dist[np.array([b.class_name for b in boxes])[:, None] != np.array([self._tracks[k].class_name for k in ids])[None]] = np.inf
        pairs = np.argwhere(dist <= self.gate_m)
        match: dict[int, str] = {}
        for i, j in pairs[np.argsort(dist[pairs[:, 0], pairs[:, 1]], kind="stable")]:
            if i not in match and ids[j] not in match.values():
                match[int(i)] = ids[j]
        return match
