"""Object velocity and short-horizon prediction in the BEV world model.

Two trackers share the same velocity math and differ only in how they match an
object to its previous-frame self:

* ``GtVelocityTracker`` - matches by the stable nuScenes ``instance_token`` (free,
  zero association error). The oracle.
* ``PredictionVelocityTracker`` - greedy nearest-by-class association in the global
  frame, for detector boxes (e.g. CenterPoint) that carry no identity. Validated
  against the oracle.

Velocity is a finite difference of global position (true ground motion) rotated
into the current ego frame; future positions are a constant-velocity extrapolation.
Both trackers reset themselves when the frame's scene changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsd.data import Pose, SurroundFrame
from fsd.object_detection import Box3D


@dataclass(frozen=True)
class TrackedObject:
    box: Box3D
    track_id: str
    velocity_ego: np.ndarray   # (vx, vy) m/s, current ego frame
    speed_mps: float
    future_xy_ego: np.ndarray  # (K, 2) constant-velocity predicted centers, current ego frame


def _global_xy(boxes: list[Box3D], ego_pose: Pose) -> np.ndarray:
    if not boxes:
        return np.empty((0, 2))
    return ego_pose.apply(np.stack([box.center_ego for box in boxes]))[:, :2]


class _VelocityTracker:
    def __init__(self, horizons_s: tuple[float, ...], min_speed_mps: float):
        self.horizons_s = horizons_s
        self.min_speed_mps = min_speed_mps
        self._scene_token: str | None = None

    def reset(self) -> None:
        raise NotImplementedError

    def _enter(self, frame: SurroundFrame) -> None:
        if frame.scene_token != self._scene_token:
            self.reset()
            self._scene_token = frame.scene_token

    def _track(self, box: Box3D, track_id: str, xy: np.ndarray, prev: tuple[np.ndarray, int] | None, frame: SurroundFrame) -> TrackedObject:
        v_ego = np.zeros(2)
        if prev is not None:
            dt = (frame.timestamp_us - prev[1]) / 1_000_000.0
            if dt > 0.0:
                v_ego = frame.ego_pose.rotation[:2, :2].T @ ((xy - prev[0]) / dt)
        speed = float(np.hypot(*v_ego))
        future = np.empty((0, 2))
        if speed >= self.min_speed_mps:
            future = np.stack([box.center_ego[:2] + v_ego * h for h in self.horizons_s])
        return TrackedObject(box, track_id, v_ego, speed, future)


class GtVelocityTracker(_VelocityTracker):
    """Finite-difference velocity over stable instance tokens (the oracle)."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5):
        super().__init__(horizons_s, min_speed_mps)
        self._prev: dict[str, tuple[np.ndarray, int]] = {}

    def reset(self) -> None:
        self._prev = {}

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        self._enter(frame)
        xys = _global_xy(boxes, frame.ego_pose)
        tracked = [
            self._track(box, box.instance_token, xy, self._prev.get(box.instance_token) if box.instance_token else None, frame)
            for box, xy in zip(boxes, xys)
        ]
        self._prev = {box.instance_token: (xy, frame.timestamp_us) for box, xy in zip(boxes, xys) if box.instance_token}
        return tracked


class PredictionVelocityTracker(_VelocityTracker):
    """Greedy nearest-by-class association for identity-free detector boxes."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5, gate_m: float = 5.0):
        super().__init__(horizons_s, min_speed_mps)
        self.gate_m = gate_m
        self.reset()

    def reset(self) -> None:
        self._ids: list[str] = []
        self._xy = np.empty((0, 2))
        self._classes: list[str] = []
        self._ts = 0
        self._next_id = 0

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        self._enter(frame)
        xys = _global_xy(boxes, frame.ego_pose)
        match = self._associate(xys, [box.class_name for box in boxes])

        tracked, ids = [], []
        for i, (box, xy) in enumerate(zip(boxes, xys)):
            prev = match.get(i)
            if prev is None:
                track_id, prev_state = f"p{self._next_id}", None
                self._next_id += 1
            else:
                track_id, prev_state = self._ids[prev], (self._xy[prev], self._ts)
            tracked.append(self._track(box, track_id, xy, prev_state, frame))
            ids.append(track_id)
        self._ids, self._xy, self._classes, self._ts = ids, xys, [b.class_name for b in boxes], frame.timestamp_us
        return tracked

    def _associate(self, xys: np.ndarray, classes: list[str]) -> dict[int, int]:
        """Map current box index -> previous track index, greedily by distance within the gate."""
        if not len(xys) or not len(self._xy):
            return {}
        dist = np.linalg.norm(xys[:, None] - self._xy[None], axis=2)
        dist[np.array(classes)[:, None] != np.array(self._classes)[None]] = np.inf
        pairs = np.argwhere(dist <= self.gate_m)
        match: dict[int, int] = {}
        used: set[int] = set()
        for i, j in pairs[np.argsort(dist[pairs[:, 0], pairs[:, 1]], kind="stable")]:
            if i not in match and j not in used:
                match[int(i)] = int(j)
                used.add(int(j))
        return match
