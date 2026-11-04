"""Compact, cached index over the nuScenes metadata tables.

sample_data.json (1.3 GB), ego_pose.json (650 MB) and sample_annotation.json
(580 MB) are too big to json.load every run and too slow to stream per query
(~40 s a scan). One streaming pass keeps only the scenes whose blobs are on
disk and only the channels we read, and pickles the result to the user cache,
so every later run starts in seconds. The cache key includes the set of
available scenes, so downloading another blob part rebuilds it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

CAMERA_CHANNELS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
LIDAR_TOP = "LIDAR_TOP"
INDEXED_CHANNELS = (*CAMERA_CHANNELS, LIDAR_TOP)
CACHE_DIR = Path.home() / ".cache" / "vision-fsd"

_SAMPLE_DATA_FIELDS = ("token", "sample_token", "timestamp", "filename", "is_key_frame", "calibrated_sensor_token", "ego_pose_token")
_ANNOTATION_FIELDS = ("token", "sample_token", "instance_token", "translation", "size", "rotation", "num_lidar_pts")


@dataclass(frozen=True)
class NuScenesIndex:
    scenes: list[dict[str, Any]]                       # available scenes, each with "location" added
    samples: dict[str, dict[str, Any]]                  # sample token -> sample record
    sweeps: dict[str, dict[str, list[dict[str, Any]]]]  # scene token -> channel -> records by time
    ego_poses: dict[str, dict[str, Any]]                # ego_pose token -> {rotation, translation}
    calibrated_sensors: dict[str, dict[str, Any]]
    annotations: dict[str, list[dict[str, Any]]]        # sample token -> annotations


def iter_json_array(path: Path, chunk_size: int = 1 << 24) -> Iterator[dict[str, Any]]:
    """Stream the objects of a top-level JSON array without loading the file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = handle.read(chunk_size).lstrip()
        if not buf.startswith("["):
            raise ValueError(f"{path} is not a JSON array")
        pos = 1
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = handle.read(chunk_size)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj


def read_table(meta_dir: Path, name: str) -> list[dict[str, Any]]:
    with (meta_dir / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_index(dataroot: Path, version: str) -> NuScenesIndex:
    meta_dir = dataroot / version
    scenes = read_table(meta_dir, "scene.json")
    samples = {s["token"]: s for s in read_table(meta_dir, "sample.json")}
    logs = {log["token"]: log for log in read_table(meta_dir, "log.json")}
    available = _available_scenes(dataroot, scenes, samples, logs)
    if not available:
        raise FileNotFoundError(
            f"No nuScenes scene under {dataroot} has its LIDAR_TOP keyframes in samples/. "
            "Extract at least one v1.0-trainval*_blobs part there."
        )

    key = hashlib.sha1(f"{dataroot.resolve()}|{version}|{','.join(sorted(s['token'] for s in available))}".encode())
    cache = CACHE_DIR / f"nuscenes-{version}-{key.hexdigest()[:12]}.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)

    print(f"Indexing nuScenes metadata for {len(available)} scenes (one-time, a few minutes)...")
    index = _build_index(meta_dir, available, samples)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def _available_scenes(dataroot: Path, scenes: list[dict], samples: dict, logs: dict) -> list[dict[str, Any]]:
    # A keyframe's timestamp is its LIDAR_TOP timestamp, so the first lidar file
    # name is derivable without the 1.3 GB sample_data table.
    lidar_dir = dataroot / "samples" / LIDAR_TOP
    available = []
    for scene in scenes:
        log = logs[scene["log_token"]]
        timestamp = samples[scene["first_sample_token"]]["timestamp"]
        if (lidar_dir / f"{log['logfile']}__{LIDAR_TOP}__{timestamp}.pcd.bin").exists():
            available.append({**scene, "location": log["location"]})
    return available


def _index_sample_data(meta_dir: Path, samples: dict, calibrated: dict) -> dict[str, dict[str, list]]:
    """scene token -> channel -> slim sample_data records sorted by time."""
    channel_of_sensor = {s["token"]: s["channel"] for s in read_table(meta_dir, "sensor.json")}
    sweeps: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for record in iter_json_array(meta_dir / "sample_data.json"):
        sample = samples.get(record["sample_token"])
        channel = channel_of_sensor[calibrated[record["calibrated_sensor_token"]]["sensor_token"]]
        if sample is not None and channel in INDEXED_CHANNELS:
            sweeps[sample["scene_token"]][channel].append({k: record[k] for k in _SAMPLE_DATA_FIELDS})
    for by_channel in sweeps.values():
        for records in by_channel.values():
            records.sort(key=lambda r: r["timestamp"])
    return {scene: dict(by_channel) for scene, by_channel in sweeps.items()}


def _build_index(meta_dir: Path, scenes: list[dict], all_samples: dict) -> NuScenesIndex:
    scene_tokens = {scene["token"] for scene in scenes}
    samples = {t: s for t, s in all_samples.items() if s["scene_token"] in scene_tokens}
    calibrated = {c["token"]: c for c in read_table(meta_dir, "calibrated_sensor.json")}
    sweeps = _index_sample_data(meta_dir, samples, calibrated)
    pose_tokens = {r["ego_pose_token"] for by_channel in sweeps.values() for records in by_channel.values() for r in records}

    ego_poses = {
        p["token"]: {"rotation": p["rotation"], "translation": p["translation"]}
        for p in iter_json_array(meta_dir / "ego_pose.json")
        if p["token"] in pose_tokens
    }
    annotations: dict[str, list] = defaultdict(list)
    for record in iter_json_array(meta_dir / "sample_annotation.json"):
        if record["sample_token"] in samples:
            annotations[record["sample_token"]].append({k: record[k] for k in _ANNOTATION_FIELDS})

    return NuScenesIndex(
        scenes=scenes,
        samples=samples,
        sweeps=sweeps,
        ego_poses=ego_poses,
        calibrated_sensors=calibrated,
        annotations=dict(annotations),
    )
