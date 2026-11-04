from __future__ import annotations

import os
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from fsd.nuscenes_index import CAMERA_CHANNELS, LIDAR_TOP, load_index, read_table

__all__ = ["CAMERA_CHANNELS", "LIDAR_TOP", "CameraFrame", "LidarFrame", "LidarSweep", "NuScenesSceneLoader", "Pose", "SurroundFrame"]

DEFAULT_DATAROOT = "C:/nuscenes"


def quaternion_to_rotation_matrix(quaternion) -> np.ndarray:
    """Convert a nuScenes wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot convert zero-norm quaternion to rotation matrix")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class Pose:
    """Rigid transform taking points from a local frame into its parent.

    Sensor calibrations map sensor -> ego; ego poses map ego -> global.
    """

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Pose:
        return cls(quaternion_to_rotation_matrix(record["rotation"]), np.asarray(record["translation"], dtype=np.float64))

    @property
    def yaw(self) -> float:
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def invert(self, points: np.ndarray) -> np.ndarray:
        return (points - self.translation) @ self.rotation

    def relative_to(self, other: Pose) -> Pose:
        """Transform from this frame into `other`'s frame (both share a parent)."""
        return Pose(other.rotation.T @ self.rotation, other.rotation.T @ (self.translation - other.translation))


@dataclass(frozen=True)
class CameraFrame:
    channel: str
    image_path: Path
    timestamp_us: int
    intrinsic: np.ndarray
    calibration: Pose
    ego_pose: Pose
    is_key_frame: bool = True


@dataclass(frozen=True)
class LidarSweep:
    path: Path
    timestamp_us: int
    calibration: Pose
    ego_pose: Pose


@dataclass(frozen=True)
class LidarFrame:
    """A LIDAR_TOP keyframe sweep followed by the sweeps before it, newest first."""

    sweeps: tuple[LidarSweep, ...]

    @property
    def path(self) -> Path:
        return self.sweeps[0].path

    @property
    def timestamp_us(self) -> int:
        return self.sweeps[0].timestamp_us

    @property
    def calibration(self) -> Pose:
        return self.sweeps[0].calibration

    @property
    def ego_pose(self) -> Pose:
        return self.sweeps[0].ego_pose


@dataclass(frozen=True)
class SurroundFrame:
    scene_token: str
    scene_name: str
    location: str
    sample_token: str
    sample_index: int
    timestamp_us: int
    # Reference ego pose: LIDAR_TOP at keyframes, the instant annotations are
    # defined at. Camera ego poses differ by tens of milliseconds.
    ego_pose: Pose
    cameras: dict[str, CameraFrame]
    is_key_frame: bool = True


class NuScenesSceneLoader:
    """Read nuScenes scenes from disk through the cached metadata index.

    Only scenes whose blobs are extracted under the dataroot are listed, so
    `scene_index` counts available scenes.
    """

    def __init__(self, dataroot: str | Path | None = None, version: str = "v1.0-trainval"):
        self.dataroot = Path(dataroot or os.getenv("NUSCENES_ROOT") or DEFAULT_DATAROOT)
        self.version = version
        self.meta_dir = self.dataroot / version
        if not self.meta_dir.exists():
            raise FileNotFoundError(f"nuScenes metadata folder not found: {self.meta_dir}")

        self.index = load_index(self.dataroot, version)
        self.scenes = self.index.scenes
        self._keyframes: dict[str, dict[str, dict]] = {}
        self._lidar_position: dict[str, int] = {}
        for by_channel in self.index.sweeps.values():
            for channel, records in by_channel.items():
                for position, record in enumerate(records):
                    if record["is_key_frame"]:
                        self._keyframes.setdefault(record["sample_token"], {})[channel] = record
                    if channel == LIDAR_TOP:
                        self._lidar_position[record["token"]] = position

    def load_table(self, name: str) -> list[dict[str, Any]]:
        return read_table(self.meta_dir, name)

    def annotations(self, sample_token: str) -> list[dict[str, Any]]:
        return self.index.annotations.get(sample_token, [])

    def scene(self, scene_index: int = 0, scene_name: str | None = None) -> dict[str, Any]:
        if scene_name is not None:
            for scene in self.scenes:
                if scene["name"] == scene_name:
                    return scene
            raise ValueError(f"Scene not available: {scene_name}")
        if not 0 <= scene_index < len(self.scenes):
            raise IndexError(f"Scene index {scene_index} out of range: {len(self.scenes)} scenes available")
        return self.scenes[scene_index]

    def samples(self, scene: dict[str, Any], start: int = 0, max_frames: int | None = None) -> list[dict[str, Any]]:
        if start < 0 or (max_frames is not None and max_frames < 0):
            raise ValueError("start and max_frames must be >= 0")
        samples = []
        token = scene["first_sample_token"]
        while token:
            samples.append(self.index.samples[token])
            token = samples[-1]["next"]
        if start >= len(samples):
            raise IndexError(f"Sample index {start} is beyond scene length {len(samples)}")
        return samples[start:] if max_frames is None else samples[start:start + max_frames]

    def future_ego_poses(self, frame: SurroundFrame, steps: int, step_s: float = 0.5) -> list[Pose]:
        """Logged global ego poses every `step_s` after `frame`, from the 20 Hz LiDAR; fewer near the scene end."""
        records = self.index.sweeps[frame.scene_token][LIDAR_TOP]
        times = np.array([r["timestamp"] for r in records])
        targets = frame.timestamp_us + step_s * 1e6 * np.arange(1, steps + 1)
        nearest = np.abs(times[:, None] - targets[targets <= times[-1]]).argmin(axis=0)
        return [self._ego_pose(records[i]) for i in nearest]

    def iter_scene_frames(
        self,
        scene_index: int = 0,
        start_sample_index: int = 0,
        max_frames: int | None = None,
        scene_name: str | None = None,
        include_lidar: bool = False,
        lidar_sweeps: int = 1,
    ) -> Iterator[tuple[SurroundFrame, LidarFrame | None]]:
        scene = self.scene(scene_index, scene_name)
        for offset, sample in enumerate(self.samples(scene, start_sample_index, max_frames)):
            records = self._keyframes[sample["token"]]
            lidar = self._lidar_frame(scene, records[LIDAR_TOP], lidar_sweeps)
            frame = self._surround_frame(scene, sample, start_sample_index + offset, records, lidar.ego_pose)
            yield frame, (lidar if include_lidar else None)

    def iter_camera_sweep_frames(
        self,
        scene_index: int = 0,
        start_frame_index: int = 0,
        max_frames: int | None = None,
        scene_name: str | None = None,
        target_channel: str = "CAM_FRONT",
        tolerance_us: int = 100_000,
    ) -> Iterator[SurroundFrame]:
        """Yield six-camera frames at every target-channel image, not just keyframes.

        Each other camera contributes its nearest image within `tolerance_us`.
        """
        if target_channel not in CAMERA_CHANNELS:
            raise ValueError(f"target_channel must be one of {CAMERA_CHANNELS}")
        scene = self.scene(scene_index, scene_name)
        by_channel = {c: self.index.sweeps[scene["token"]][c] for c in CAMERA_CHANNELS}
        groups = _group_by_nearest(by_channel, target_channel, tolerance_us)[start_frame_index:]
        for offset, records in enumerate(groups[:max_frames] if max_frames is not None else groups):
            target = records[target_channel]
            yield self._surround_frame(
                scene,
                self.index.samples[target["sample_token"]],
                start_frame_index + offset,
                records,
                self._ego_pose(target),
                timestamp_us=target["timestamp"],
                is_key_frame=all(r["is_key_frame"] for r in records.values()),
            )

    def iter_lidar_sweep_frames(
        self, scene_index: int = 0, max_frames: int | None = None, scene_name: str | None = None, lidar_sweeps: int = 1, stride: int = 1
    ) -> Iterator[tuple[SurroundFrame, LidarFrame]]:
        """Yield every `stride`-th LIDAR_TOP sweep (20 Hz / stride) as a frame without cameras, with its history."""
        scene = self.scene(scene_index, scene_name)
        for i, record in enumerate(self.index.sweeps[scene["token"]][LIDAR_TOP][::stride][:max_frames]):
            lidar = self._lidar_frame(scene, record, lidar_sweeps)
            frame = SurroundFrame(scene["token"], scene["name"], scene["location"], record["sample_token"], i, record["timestamp"], lidar.ego_pose, {}, record["is_key_frame"])
            yield frame, lidar

    def _ego_pose(self, record: dict[str, Any]) -> Pose:
        return Pose.from_record(self.index.ego_poses[record["ego_pose_token"]])

    def _calibration(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.index.calibrated_sensors[record["calibrated_sensor_token"]]

    def _lidar_frame(self, scene: dict[str, Any], record: dict[str, Any], n_sweeps: int) -> LidarFrame:
        records = self.index.sweeps[scene["token"]][LIDAR_TOP]
        position = self._lidar_position[record["token"]]
        history = records[max(0, position - n_sweeps + 1): position + 1][::-1]
        return LidarFrame(tuple(
            LidarSweep(
                path=self.dataroot / r["filename"],
                timestamp_us=r["timestamp"],
                calibration=Pose.from_record(self._calibration(r)),
                ego_pose=self._ego_pose(r),
            )
            for r in history
        ))

    def _surround_frame(
        self,
        scene: dict[str, Any],
        sample: dict[str, Any],
        sample_index: int,
        records: dict[str, dict[str, Any]],
        ego_pose: Pose,
        timestamp_us: int | None = None,
        is_key_frame: bool = True,
    ) -> SurroundFrame:
        return SurroundFrame(
            scene_token=scene["token"],
            scene_name=scene["name"],
            location=scene["location"],
            sample_token=sample["token"],
            sample_index=sample_index,
            timestamp_us=sample["timestamp"] if timestamp_us is None else timestamp_us,
            ego_pose=ego_pose,
            cameras={channel: self._camera_frame(channel, records[channel]) for channel in CAMERA_CHANNELS},
            is_key_frame=is_key_frame,
        )

    def _camera_frame(self, channel: str, record: dict[str, Any]) -> CameraFrame:
        calibration = self._calibration(record)
        return CameraFrame(
            channel=channel,
            image_path=self.dataroot / record["filename"],
            timestamp_us=record["timestamp"],
            intrinsic=np.asarray(calibration["camera_intrinsic"], dtype=np.float64),
            calibration=Pose.from_record(calibration),
            ego_pose=self._ego_pose(record),
            is_key_frame=record["is_key_frame"],
        )


def _group_by_nearest(
    records_by_channel: dict[str, list[dict[str, Any]]],
    target_channel: str,
    tolerance_us: int,
) -> list[dict[str, dict[str, Any]]]:
    timestamps = {c: [r["timestamp"] for r in records] for c, records in records_by_channel.items()}
    groups = []
    for target in records_by_channel[target_channel]:
        group = {}
        for channel, records in records_by_channel.items():
            i = bisect_left(timestamps[channel], target["timestamp"])
            nearest = min(records[max(0, i - 1): i + 1], key=lambda r: abs(r["timestamp"] - target["timestamp"]))
            if abs(nearest["timestamp"] - target["timestamp"]) > tolerance_us:
                break
            group[channel] = nearest
        else:
            groups.append(group)
    return groups
