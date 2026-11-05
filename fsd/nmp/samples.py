"""Training samples: one per keyframe, cached as compressed npz.

A sample holds the network input in sparse form (occupied voxel indices for 10
sweeps, plus map masks), the ego's future over the next T keyframes, and GT
boxes now and at those keyframes, all in the keyframe's ego frame.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from fsd.data import LidarFrame, LidarSweep, NuScenesSceneLoader, SurroundFrame
from fsd.ego_motion import wrap_angle
from fsd.lidar_projection import lidar_points_ego
from fsd.nmp.config import NmpConfig
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.object_detection import DETECTION_CLASSES, NuScenesAnnotationLoader, box_from_global

CLASS_NAMES = tuple(sorted(set(DETECTION_CLASSES.values())))
VAL_SCENES = Path(__file__).with_name("val_scenes.txt")


@dataclass(frozen=True)
class NmpSample:
    token: str
    scene: str
    voxels: np.ndarray      # uint32 flat indices into (n_sweeps * z_bins, H, W)
    map_masks: np.ndarray   # uint8 (M, H, W)
    ego_future: np.ndarray  # float32 (T, 3): x, y, yaw
    v0: float
    k0: float
    boxes: np.ndarray       # float32 (K, T+1, 5): x, y, w, l, yaw; step 0 is now
    box_valid: np.ndarray   # bool (K, T+1)
    box_class: np.ndarray   # int8 (K,) index into CLASS_NAMES
    box_points: np.ndarray  # int32 (K,) LiDAR returns in the step-0 box; 0 when unseen or absent


def voxelize(points: np.ndarray, sweep: np.ndarray, cfg: NmpConfig) -> np.ndarray:
    """Unique flat indices of occupied (sweep * z_bins + z_bin, row, col) voxels."""
    rows, cols, inside = cfg.grid.cells(points[:, :2])
    zbin = np.floor((points[:, 2] - cfg.z_range[0]) / cfg.z_resolution).astype(np.int64)
    keep = inside & (zbin >= 0) & (zbin < cfg.z_bins)
    h, w = cfg.grid.shape
    channel = sweep[keep] * cfg.z_bins + zbin[keep]
    return np.unique((channel * h + rows[keep]) * w + cols[keep]).astype(np.uint32)


def dense_input(voxels: np.ndarray, map_masks: np.ndarray, cfg: NmpConfig) -> np.ndarray:
    """(in_channels, H, W) uint8 network input from sparse voxel indices and map masks."""
    h, w = cfg.grid.shape
    dense = np.zeros(cfg.n_sweeps * cfg.z_bins * h * w, np.uint8)
    dense[voxels] = 1
    return np.concatenate([dense.reshape(-1, h, w), map_masks])


def ego_state(lidar: LidarFrame) -> tuple[float, float]:
    """(speed m/s, curvature 1/m): speed over the last ~0.1 s, curvature over the shortest recent span of 0.5 m."""
    if len(lidar.sweeps) < 2:
        return 0.0, 0.0
    span = min(2, len(lidar.sweeps) - 1)
    now, before = lidar.sweeps[0], lidar.sweeps[span]
    dt = (now.timestamp_us - before.timestamp_us) * 1e-6
    speed = _travelled(now, before) / dt if dt > 0 else 0.0
    # Below half a metre the heading change is dominated by pose noise, so slow turns reach further back;
    # never shorter than the ~0.1 s speed span, where one sweep's pose noise would dominate at speed.
    for past in lidar.sweeps[span:]:
        dist = _travelled(now, past)
        if dist > 0.5:
            return speed, wrap_angle(now.ego_pose.yaw - past.ego_pose.yaw) / dist
    return speed, 0.0


def _travelled(now: LidarSweep, past: LidarSweep) -> float:
    return float(np.hypot(*(now.ego_pose.translation - past.ego_pose.translation)[:2]))


def sweep_points(lidar: LidarFrame, cfg: NmpConfig) -> tuple[np.ndarray, np.ndarray]:
    selected = LidarFrame(lidar.sweeps[:: cfg.sweep_stride][: cfg.n_sweeps])
    points = lidar_points_ego(selected)
    ages = np.array([(selected.timestamp_us - s.timestamp_us) * 1e-6 for s in selected.sweeps], dtype=np.float32)
    return points, np.searchsorted(ages, points[:, 3])


def _future_boxes(
    frames: list[SurroundFrame],
    annotations: list[list[dict]],
    category_of_instance: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(K, T+1, 5) boxes in frames[0]'s ego frame for every instance seen at any step, with validity, class and step-0 LiDAR count."""
    rows: dict[str, int] = {}
    entries = []
    for step, records in enumerate(annotations):
        for record in records:
            name = DETECTION_CLASSES.get(category_of_instance[record["instance_token"]])
            if name is None:
                continue
            row = rows.setdefault(record["instance_token"], len(rows))
            box = box_from_global(record, frames[0].ego_pose, name)
            entries.append((row, step, [*box.center_ego[:2], *box.size[:2], box.yaw_ego], CLASS_NAMES.index(name), box.num_lidar_pts))
    boxes = np.zeros((len(rows), len(frames), 5), np.float32)
    valid = np.zeros((len(rows), len(frames)), bool)
    classes = np.zeros(len(rows), np.int8)
    points = np.zeros(len(rows), np.int32)
    for row, step, values, cls, count in entries:
        boxes[row, step], valid[row, step], classes[row] = values, True, cls
        if step == 0:
            points[row] = count
    return boxes, valid, classes, points


def build_scene_samples(
    loader: NuScenesSceneLoader,
    map_renderer: NuScenesMapRenderer,
    annotations: NuScenesAnnotationLoader,
    scene_name: str,
    cfg: NmpConfig,
) -> list[NmpSample]:
    frames = list(loader.iter_scene_frames(scene_name=scene_name, include_lidar=True, lidar_sweeps=cfg.history_sweeps))
    samples = []
    for i in range(len(frames) - cfg.horizon):
        lidar = frames[i][1]
        if len(lidar.sweeps) >= cfg.history_sweeps:
            window = [frame for frame, _ in frames[i : i + cfg.horizon + 1]]
            samples.append(_sample(loader, map_renderer, annotations, window, lidar, cfg))
    return samples


def _sample(
    loader: NuScenesSceneLoader,
    map_renderer: NuScenesMapRenderer,
    annotations: NuScenesAnnotationLoader,
    window: list[SurroundFrame],
    lidar: LidarFrame,
    cfg: NmpConfig,
) -> NmpSample:
    frame = window[0]
    future = [f.ego_pose.relative_to(frame.ego_pose) for f in window[1:]]
    points, sweep = sweep_points(lidar, cfg)
    boxes, valid, classes, box_points = _future_boxes(window, [loader.annotations(f.sample_token) for f in window], annotations.category_of_instance)
    v0, k0 = ego_state(lidar)
    return NmpSample(
        token=frame.sample_token,
        scene=frame.scene_name,
        voxels=voxelize(points, sweep, cfg),
        map_masks=map_renderer.layer_masks(frame, cfg.grid, cfg.map_layers),
        ego_future=np.array([[p.translation[0], p.translation[1], p.yaw] for p in future], np.float32),
        v0=v0,
        k0=k0,
        boxes=boxes,
        box_valid=valid,
        box_class=classes,
        box_points=box_points,
    )


def split_scenes(loader: NuScenesSceneLoader) -> dict[str, list[str]]:
    """Official nuScenes val scenes vs. every other available scene."""
    val = set(VAL_SCENES.read_text().split())
    names = [scene["name"] for scene in loader.scenes]
    return {"train": [n for n in names if n not in val], "val": [n for n in names if n in val]}


def save_sample(sample: NmpSample, path: Path) -> None:
    np.savez_compressed(path, **{f.name: getattr(sample, f.name) for f in fields(NmpSample)})


def load_sample(path: Path) -> NmpSample:
    with np.load(path) as data:
        missing = [f.name for f in fields(NmpSample) if f.name not in data.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}: the cache predates them; rebuild it with `python -m fsd.nmp.train cache`")
        values = {f.name: data[f.name] for f in fields(NmpSample)}
    for name, cast in (("token", str), ("scene", str), ("v0", float), ("k0", float)):
        values[name] = cast(values[name])
    return NmpSample(**values)
