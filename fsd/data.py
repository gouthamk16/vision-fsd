from __future__ import annotations

import json
import os
f`rom bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAMERA_CHANNELS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
LIDAR_TOP = "LIDAR_TOP"


@dataclass(frozen=True)
class CameraFrame:
    channel: str
    image_path: Path
    timestamp_us: int
    sample_data_token: str
    calibrated_sensor: dict[str, Any]
    ego_pose: dict[str, Any]
    is_key_frame: bool = True

    @property
    def camera_intrinsic(self) -> list[list[float]]:
        return self.calibrated_sensor["camera_intrinsic"]

    @property
    def sensor_translation(self) -> list[float]:
        return self.calibrated_sensor["translation"]

    @property
    def sensor_rotation(self) -> list[float]:
        return self.calibrated_sensor["rotation"]


@dataclass(frozen=True)
class LidarFrame:
    channel: str
    pointcloud_path: Path
    timestamp_us: int
    sample_data_token: str
    calibrated_sensor: dict[str, Any]
    ego_pose: dict[str, Any]

    @property
    def sensor_translation(self) -> list[float]:
        return self.calibrated_sensor["translation"]

    @property
    def sensor_rotation(self) -> list[float]:
        return self.calibrated_sensor["rotation"]


@dataclass(frozen=True)
class SurroundFrame:
    scene_token: str
    scene_name: str
    sample_token: str
    sample_index: int
    timestamp_us: int
    cameras: dict[str, CameraFrame]
    is_key_frame: bool = True

    @property
    def ego_pose(self) -> dict[str, Any]:
        if "CAM_FRONT" in self.cameras:
            return self.cameras["CAM_FRONT"].ego_pose
        first_camera = next(iter(self.cameras.values()))
        return first_camera.ego_pose


class NuScenesSceneLoader:
    """Read synchronized nuScenes surround-camera samples directly from disk.

    The loader treats nuScenes as an external read-only dataset. It keeps paths
    pointing at the dataset root and does not copy image or metadata files.
    """

    def __init__(self, dataroot: str | Path | None = None, version: str = "v1.0-trainval"):
        root = dataroot or os.getenv("NUSCENES_ROOT") or "D:/nuscenes"
        self.dataroot = Path(root)
        self.version = version
        self.meta_dir = self.dataroot / version

        if not self.dataroot.exists():
            raise FileNotFoundError(f"nuScenes root not found: {self.dataroot}")
        if not self.meta_dir.exists():
            raise FileNotFoundError(f"nuScenes metadata folder not found: {self.meta_dir}")

        self.scenes = self._load_table("scene.json")
        self._samples = self._index_by_token(self._load_table("sample.json"))
        self._sensors = self._index_by_token(self._load_table("sensor.json"))
        self._calibrated_sensors = self._index_by_token(self._load_table("calibrated_sensor.json"))

        self._sample_data_cache: dict[str, dict[str, Any]] = {}
        self._sample_camera_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._ego_pose_cache: dict[str, dict[str, Any]] = {}

    def list_scenes(self) -> list[dict[str, Any]]:
        return self.scenes

    def get_surround_frame(
        self,
        scene_index: int = 0,
        sample_index: int = 0,
        scene_name: str | None = None,
    ) -> SurroundFrame:
        scene = self._select_scene(scene_index=scene_index, scene_name=scene_name)
        sample = self._sample_at(scene, sample_index)

        sample_data_by_channel = self._get_sample_camera_data(sample["token"])
        sample_data = {
            record["token"]: record
            for record in sample_data_by_channel.values()
        }

        ego_pose_tokens = {record["ego_pose_token"] for record in sample_data.values()}
        ego_poses = self._get_ego_poses(ego_pose_tokens)

        cameras: dict[str, CameraFrame] = {}
        for channel in CAMERA_CHANNELS:
            record = sample_data_by_channel.get(channel)
            if record is None:
                continue
            token = record["token"]
            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            sensor = self._sensors[calibrated["sensor_token"]]
            if sensor["channel"] != channel:
                raise ValueError(f"Expected {channel}, got {sensor['channel']} for {token}")

            image_path = self.dataroot / record["filename"]
            cameras[channel] = CameraFrame(
                channel=channel,
                image_path=image_path,
                timestamp_us=record["timestamp"],
                sample_data_token=token,
                calibrated_sensor=calibrated,
                ego_pose=ego_poses[record["ego_pose_token"]],
            )

        return SurroundFrame(
            scene_token=scene["token"],
            scene_name=scene["name"],
            sample_token=sample["token"],
            sample_index=sample_index,
            timestamp_us=sample["timestamp"],
            cameras=cameras,
        )

    def iter_scene_frames(
        self,
        scene_index: int = 0,
        start_sample_index: int = 0,
        max_frames: int | None = None,
        scene_name: str | None = None,
        include_lidar: bool = False,
    ):
        """Yield scene frames after prefetching required metadata in one pass."""
        scene = self._select_scene(scene_index=scene_index, scene_name=scene_name)
        samples = self._sample_sequence(scene, start_sample_index, max_frames)
        if not samples:
            return

        channels = set(CAMERA_CHANNELS)
        if include_lidar:
            channels.add(LIDAR_TOP)

        sample_tokens = {sample["token"] for _, sample in samples}
        sample_records = self._get_scene_sample_channel_data(sample_tokens, channels)

        ego_pose_tokens = {
            record["ego_pose_token"]
            for records_by_channel in sample_records.values()
            for record in records_by_channel.values()
        }
        ego_poses = self._get_ego_poses(ego_pose_tokens)

        for absolute_index, sample in samples:
            records_by_channel = sample_records[sample["token"]]
            frame = self._make_surround_frame(
                scene=scene,
                sample=sample,
                sample_index=absolute_index,
                records_by_channel=records_by_channel,
                ego_poses=ego_poses,
            )
            lidar = None
            if include_lidar:
                lidar_record = records_by_channel[LIDAR_TOP]
                lidar = self._make_lidar_frame(lidar_record, ego_poses)
            yield frame, lidar

    def iter_camera_sweep_frames(
        self,
        scene_index: int = 0,
        start_frame_index: int = 0,
        max_frames: int | None = None,
        scene_name: str | None = None,
        target_channel: str = "CAM_FRONT",
        tolerance_us: int = 100_000,
    ):
        """Yield smoother six-camera frames grouped from camera sweeps.

        The target channel supplies the timeline. For each target timestamp, the
        nearest image from every other camera is selected if it is within the
        tolerance. This produces real intermediate camera frames instead of
        speeding up sparse key samples.
        """
        if target_channel not in CAMERA_CHANNELS:
            raise ValueError(f"target_channel must be one of {CAMERA_CHANNELS}")
        if start_frame_index < 0:
            raise ValueError("start_frame_index must be >= 0")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        if max_frames == 0:
            return

        scene = self._select_scene(scene_index=scene_index, scene_name=scene_name)
        sample_tokens = {sample["token"] for _, sample in self._sample_sequence(scene, 0, None)}
        records_by_channel = self._get_scene_camera_records(
            sample_tokens=sample_tokens,
            include_non_key=True,
        )
        groups = self._group_camera_records_by_target_channel(
            records_by_channel=records_by_channel,
            target_channel=target_channel,
            tolerance_us=tolerance_us,
        )
        groups = groups[start_frame_index:]
        if max_frames is not None:
            groups = groups[:max_frames]
        if not groups:
            return

        ego_pose_tokens = {
            record["ego_pose_token"]
            for records_by_channel in groups
            for record in records_by_channel.values()
        }
        ego_poses = self._get_ego_poses(ego_pose_tokens)

        for relative_index, records in enumerate(groups):
            target_record = records[target_channel]
            yield self._make_surround_frame(
                scene=scene,
                sample=self._samples[target_record["sample_token"]],
                sample_index=start_frame_index + relative_index,
                records_by_channel=records,
                ego_poses=ego_poses,
                timestamp_us=target_record["timestamp"],
                is_key_frame=all(record.get("is_key_frame", False) for record in records.values()),
            )

    def get_lidar_frame(
        self,
        scene_index: int = 0,
        sample_index: int = 0,
        scene_name: str | None = None,
        channel: str = LIDAR_TOP,
    ) -> LidarFrame:
        scene = self._select_scene(scene_index=scene_index, scene_name=scene_name)
        sample = self._sample_at(scene, sample_index)
        record = self._get_sample_channel_data(sample["token"], channel)
        ego_pose = self._get_ego_poses({record["ego_pose_token"]})[record["ego_pose_token"]]
        return self._make_lidar_frame(record, {record["ego_pose_token"]: ego_pose})

    def _load_table(self, name: str) -> list[dict[str, Any]]:
        with (self.meta_dir / name).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _index_by_token(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {record["token"]: record for record in records}

    def _select_scene(self, scene_index: int, scene_name: str | None) -> dict[str, Any]:
        if scene_name is not None:
            for scene in self.scenes:
                if scene["name"] == scene_name:
                    return scene
            raise ValueError(f"Scene not found: {scene_name}")
        try:
            return self.scenes[scene_index]
        except IndexError as exc:
            raise IndexError(f"Scene index out of range: {scene_index}") from exc

    def _sample_at(self, scene: dict[str, Any], sample_index: int) -> dict[str, Any]:
        token = scene["first_sample_token"]
        sample = self._samples[token]
        for _ in range(sample_index):
            token = sample["next"]
            if not token:
                raise IndexError(
                    f"Sample index {sample_index} is beyond scene length {scene['nbr_samples']}"
                )
            sample = self._samples[token]
        return sample

    def _sample_sequence(
        self,
        scene: dict[str, Any],
        start_sample_index: int,
        max_frames: int | None,
    ) -> list[tuple[int, dict[str, Any]]]:
        if start_sample_index < 0:
            raise ValueError("start_sample_index must be >= 0")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        if max_frames == 0:
            return []

        sample = self._sample_at(scene, start_sample_index)
        samples = [(start_sample_index, sample)]
        while sample["next"]:
            if max_frames is not None and len(samples) >= max_frames:
                break
            sample = self._samples[sample["next"]]
            samples.append((start_sample_index + len(samples), sample))
        return samples

    def _make_surround_frame(
        self,
        scene: dict[str, Any],
        sample: dict[str, Any],
        sample_index: int,
        records_by_channel: dict[str, dict[str, Any]],
        ego_poses: dict[str, dict[str, Any]],
        timestamp_us: int | None = None,
        is_key_frame: bool = True,
    ) -> SurroundFrame:
        cameras: dict[str, CameraFrame] = {}
        for channel in CAMERA_CHANNELS:
            record = records_by_channel.get(channel)
            if record is None:
                continue

            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            cameras[channel] = CameraFrame(
                channel=channel,
                image_path=self.dataroot / record["filename"],
                timestamp_us=record["timestamp"],
                sample_data_token=record["token"],
                calibrated_sensor=calibrated,
                ego_pose=ego_poses[record["ego_pose_token"]],
                is_key_frame=record.get("is_key_frame", True),
            )

        return SurroundFrame(
            scene_token=scene["token"],
            scene_name=scene["name"],
            sample_token=sample["token"],
            sample_index=sample_index,
            timestamp_us=sample["timestamp"] if timestamp_us is None else timestamp_us,
            cameras=cameras,
            is_key_frame=is_key_frame,
        )

    def _make_lidar_frame(
        self,
        record: dict[str, Any],
        ego_poses: dict[str, dict[str, Any]],
    ) -> LidarFrame:
        calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
        return LidarFrame(
            channel=LIDAR_TOP,
            pointcloud_path=self.dataroot / record["filename"],
            timestamp_us=record["timestamp"],
            sample_data_token=record["token"],
            calibrated_sensor=calibrated,
            ego_pose=ego_poses[record["ego_pose_token"]],
        )

    def _get_sample_data(self, tokens: list[str]) -> dict[str, dict[str, Any]]:
        missing = set(tokens) - self._sample_data_cache.keys()
        if missing:
            self._sample_data_cache.update(
                self._records_by_token_from_large_table("sample_data.json", missing)
            )
        return {token: self._sample_data_cache[token] for token in tokens}

    def _get_sample_camera_data(self, sample_token: str) -> dict[str, dict[str, Any]]:
        cached = self._sample_camera_cache.get(sample_token)
        if cached is not None:
            return cached

        cameras: dict[str, dict[str, Any]] = {}
        table_path = self.meta_dir / "sample_data.json"
        for record in _iter_json_objects(table_path):
            if record.get("sample_token") != sample_token:
                continue
            if not record.get("is_key_frame", False):
                continue

            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            sensor = self._sensors[calibrated["sensor_token"]]
            channel = sensor["channel"]
            if channel not in CAMERA_CHANNELS:
                continue

            cameras[channel] = record
            self._sample_data_cache[record["token"]] = record
            if len(cameras) == len(CAMERA_CHANNELS):
                self._sample_camera_cache[sample_token] = cameras
                return cameras

        missing = set(CAMERA_CHANNELS) - cameras.keys()
        raise KeyError(f"Missing camera sample_data for sample {sample_token}: {sorted(missing)}")

    def _get_sample_channel_data(self, sample_token: str, channel: str) -> dict[str, Any]:
        table_path = self.meta_dir / "sample_data.json"
        for record in _iter_json_objects(table_path):
            if record.get("sample_token") != sample_token:
                continue
            if not record.get("is_key_frame", False):
                continue

            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            sensor = self._sensors[calibrated["sensor_token"]]
            if sensor["channel"] != channel:
                continue

            self._sample_data_cache[record["token"]] = record
            return record

        raise KeyError(f"Missing {channel} sample_data for sample {sample_token}")

    def _get_scene_sample_channel_data(
        self,
        sample_tokens: set[str],
        channels: set[str],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        records_by_sample = {token: {} for token in sample_tokens}
        table_path = self.meta_dir / "sample_data.json"
        expected_records = len(sample_tokens) * len(channels)

        for record in _iter_json_objects(table_path):
            sample_token = record.get("sample_token")
            if sample_token not in sample_tokens:
                continue
            if not record.get("is_key_frame", False):
                continue

            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            sensor = self._sensors[calibrated["sensor_token"]]
            channel = sensor["channel"]
            if channel not in channels:
                continue

            records_by_sample[sample_token][channel] = record
            self._sample_data_cache[record["token"]] = record
            if sum(len(records) for records in records_by_sample.values()) == expected_records:
                break

        missing = {
            sample_token: sorted(channels - records.keys())
            for sample_token, records in records_by_sample.items()
            if channels - records.keys()
        }
        if missing:
            preview = list(missing.items())[:3]
            raise KeyError(f"Missing scene sample_data channels: {preview}")

        return records_by_sample

    def _get_scene_camera_records(
        self,
        sample_tokens: set[str],
        include_non_key: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        records_by_channel: dict[str, list[dict[str, Any]]] = {
            channel: []
            for channel in CAMERA_CHANNELS
        }
        table_path = self.meta_dir / "sample_data.json"

        for record in _iter_json_objects(table_path):
            if record.get("sample_token") not in sample_tokens:
                continue
            if not include_non_key and not record.get("is_key_frame", False):
                continue

            calibrated = self._calibrated_sensors[record["calibrated_sensor_token"]]
            sensor = self._sensors[calibrated["sensor_token"]]
            channel = sensor["channel"]
            if channel not in records_by_channel:
                continue

            records_by_channel[channel].append(record)
            self._sample_data_cache[record["token"]] = record

        missing = [channel for channel, records in records_by_channel.items() if not records]
        if missing:
            raise KeyError(f"Missing camera records for channels: {missing}")

        for records in records_by_channel.values():
            records.sort(key=lambda record: record["timestamp"])
        return records_by_channel

    @staticmethod
    def _group_camera_records_by_target_channel(
        records_by_channel: dict[str, list[dict[str, Any]]],
        target_channel: str,
        tolerance_us: int,
    ) -> list[dict[str, dict[str, Any]]]:
        timestamps_by_channel = {
            channel: [record["timestamp"] for record in records]
            for channel, records in records_by_channel.items()
        }
        groups: list[dict[str, dict[str, Any]]] = []

        for target_record in records_by_channel[target_channel]:
            target_timestamp = target_record["timestamp"]
            group: dict[str, dict[str, Any]] = {}
            valid = True

            for channel, records in records_by_channel.items():
                timestamps = timestamps_by_channel[channel]
                index = bisect_left(timestamps, target_timestamp)
                candidates = []
                if index < len(records):
                    candidates.append(records[index])
                if index > 0:
                    candidates.append(records[index - 1])
                if not candidates:
                    valid = False
                    break

                nearest = min(
                    candidates,
                    key=lambda record: abs(record["timestamp"] - target_timestamp),
                )
                if abs(nearest["timestamp"] - target_timestamp) > tolerance_us:
                    valid = False
                    break
                group[channel] = nearest

            if valid:
                groups.append(group)

        return groups

    def _get_ego_poses(self, tokens: set[str]) -> dict[str, dict[str, Any]]:
        missing = tokens - self._ego_pose_cache.keys()
        if missing:
            self._ego_pose_cache.update(
                self._records_by_token_from_large_table("ego_pose.json", missing)
            )
        return {token: self._ego_pose_cache[token] for token in tokens}

    def _records_by_token_from_large_table(
        self,
        table_name: str,
        tokens: set[str],
    ) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        table_path = self.meta_dir / table_name
        for record in _iter_json_objects(table_path):
            token = record.get("token")
            if token in tokens:
                found[token] = record
                if len(found) == len(tokens):
                    return found

        missing = tokens - found.keys()
        raise KeyError(f"Missing {len(missing)} token(s) in {table_path}: {sorted(missing)[:3]}")


def _iter_json_objects(path: Path):
    """Yield objects from a pretty-printed JSON array without loading it all."""
    with path.open("r", encoding="utf-8") as handle:
        collecting = False
        depth = 0
        lines: list[str] = []

        for raw_line in handle:
            stripped = raw_line.strip()
            if not collecting:
                if stripped.startswith("{"):
                    collecting = True
                    lines = [raw_line]
                    depth = raw_line.count("{") - raw_line.count("}")
                continue

            lines.append(raw_line)
            depth += raw_line.count("{") - raw_line.count("}")
            if depth == 0:
                text = "".join(lines).strip()
                if text.endswith(","):
                    text = text[:-1]
                yield json.loads(text)
                collecting = False
                lines = []
