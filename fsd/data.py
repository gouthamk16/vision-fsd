from __future__ import annotations

import json
import os
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


@dataclass(frozen=True)
class CameraFrame:
    channel: str
    image_path: Path
    timestamp_us: int
    sample_data_token: str
    calibrated_sensor: dict[str, Any]
    ego_pose: dict[str, Any]

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
class SurroundFrame:
    scene_token: str
    scene_name: str
    sample_token: str
    sample_index: int
    timestamp_us: int
    cameras: dict[str, CameraFrame]

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
