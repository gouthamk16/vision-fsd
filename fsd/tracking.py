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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsd.data import SurroundFrame
from fsd.lidar_projection import quaternion_to_rotation_matrix, transform_points
from fsd.object_detection import Box3D


@dataclass(frozen=True)
class TrackedObject:
    box: Box3D
    track_id: str
    velocity_ego: np.ndarray   # (vx, vy) m/s, current ego frame
    speed_mps: float
    future_xy_ego: np.ndarray  # (K, 2) constant-velocity predicted centers, current ego frame


def _global_xy(box: Box3D, ego_pose: dict) -> np.ndarray:
    """Object centre in the global frame (exact inverse of how center_ego was built)."""
    return transform_points(
        box.center_ego.reshape(1, 3), ego_pose["rotation"], ego_pose["translation"]
    )[0, :2]


def _make_tracked(
    box: Box3D,
    track_id: str,
    global_xy: np.ndarray,
    prev: tuple[np.ndarray, int] | None,
    ts: int,
    rot_ego: np.ndarray,
    horizons_s: tuple[float, ...],
    min_speed_mps: float,
) -> TrackedObject:
    v_ego = np.zeros(2)
    speed = 0.0
    if prev is not None:
        prev_xy, prev_ts = prev
        dt = (ts - prev_ts) / 1_000_000.0
        if dt > 0.0:
            v_ego = rot_ego.T @ ((global_xy - prev_xy) / dt)
            speed = float(np.hypot(v_ego[0], v_ego[1]))
    if speed >= min_speed_mps:
        future = np.stack([box.center_ego[:2] + v_ego * h for h in horizons_s])
    else:
        future = np.empty((0, 2))
    return TrackedObject(box=box, track_id=track_id, velocity_ego=v_ego, speed_mps=speed, future_xy_ego=future)


class GtVelocityTracker:
    """Finite-difference velocity over stable instance tokens (the oracle)."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5):
        self.horizons_s = horizons_s
        self.min_speed_mps = min_speed_mps
        self._prev: dict[str, tuple[np.ndarray, int]] = {}

    def reset(self) -> None:
        self._prev.clear()

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        ego_pose = frame.ego_pose
        rot_ego = quaternion_to_rotation_matrix(ego_pose["rotation"])[:2, :2]
        ts = frame.timestamp_us

        tracked: list[TrackedObject] = []
        current: dict[str, tuple[np.ndarray, int]] = {}
        for box in boxes:
            token = box.instance_token
            global_xy = _global_xy(box, ego_pose)
            if token:
                current[token] = (global_xy, ts)
            prev = self._prev.get(token) if token else None
            tracked.append(
                _make_tracked(box, token, global_xy, prev, ts, rot_ego, self.horizons_s, self.min_speed_mps)
            )
        self._prev = current
        return tracked


class PredictionVelocityTracker:
    """Greedy nearest-by-class association for identity-free detector boxes."""

    def __init__(
        self,
        horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0),
        min_speed_mps: float = 0.5,
        gate_m: float = 5.0,
    ):
        self.horizons_s = horizons_s
        self.min_speed_mps = min_speed_mps
        self.gate_m = gate_m
        self._tracks: dict[str, tuple[np.ndarray, int, str]] = {}  # id -> (global_xy, ts, class)
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        ego_pose = frame.ego_pose
        rot_ego = quaternion_to_rotation_matrix(ego_pose["rotation"])[:2, :2]
        ts = frame.timestamp_us
        cur_xy = [_global_xy(box, ego_pose) for box in boxes]

        # Candidate matches: same class within the gate, assigned greedily by distance.
        candidates: list[tuple[float, int, str]] = []
        for i, box in enumerate(boxes):
            for tid, (gxy, _, cls) in self._tracks.items():
                if cls != box.class_name:
                    continue
                dist = float(np.hypot(*(cur_xy[i] - gxy)))
                if dist <= self.gate_m:
                    candidates.append((dist, i, tid))
        candidates.sort(key=lambda c: c[0])

        box_to_tid: dict[int, str] = {}
        used_tids: set[str] = set()
        for dist, i, tid in candidates:
            if i in box_to_tid or tid in used_tids:
                continue
            box_to_tid[i] = tid
            used_tids.add(tid)

        tracked: list[TrackedObject] = []
        new_tracks: dict[str, tuple[np.ndarray, int, str]] = {}
        for i, box in enumerate(boxes):
            tid = box_to_tid.get(i)
            prev = self._tracks.get(tid) if tid is not None else None
            prev_state = (prev[0], prev[1]) if prev is not None else None
            if tid is None:
                tid = f"p{self._next_id}"
                self._next_id += 1
            tracked.append(
                _make_tracked(box, tid, cur_xy[i], prev_state, ts, rot_ego, self.horizons_s, self.min_speed_mps)
            )
            new_tracks[tid] = (cur_xy[i], ts, box.class_name)
        self._tracks = new_tracks
        return tracked
