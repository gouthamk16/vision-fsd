"""2.5D LiDAR BEV tensor: per-cell geometric channels in the ego frame.

Turns a raw LiDAR sweep into a multi-channel top-down grid. Every channel is
computed from the point cloud alone (no labels, no map) so it is deployable on a
real vehicle. The tensor is the data structure downstream modules (planning,
lane fitting, the object layer) read instead of re-rasterizing points.

Channels (per cell):
  density       point count (proxy for surface support / confidence)
  max_height    tallest return  -> obstacle tops, overhangs
  min_height    lowest return   -> ground level
  mean_height   average return
  height_range  max - min       -> vertical extent (flat road ~0, cars/walls large)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fsd.bev import lidar_points_to_ego
from fsd.data import LidarFrame, NuScenesSceneLoader, SurroundFrame
from fsd.lidar_projection import load_lidar_points


CHANNEL_NAMES = ("density", "max_height", "min_height", "mean_height", "height_range")


@dataclass(frozen=True)
class BevTensor:
    """Per-cell geometric channels for one ego-frame BEV grid."""

    density: np.ndarray
    max_height: np.ndarray
    min_height: np.ndarray
    mean_height: np.ndarray
    height_range: np.ndarray
    occupied: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float

    def stack(self) -> np.ndarray:
        """Return an (H, W, 5) float32 array in CHANNEL_NAMES order."""
        return np.stack(
            [self.density, self.max_height, self.min_height, self.mean_height, self.height_range],
            axis=-1,
        ).astype(np.float32)


def compute_bev_height_channels(
    points_ego: np.ndarray,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-3.0, 5.0),
    resolution: float = 0.25,
) -> BevTensor:
    """Rasterize ego-frame LiDAR points into per-cell height statistics."""
    h = int(round((x_range[1] - x_range[0]) / resolution))
    w = int(round((y_range[1] - y_range[0]) / resolution))

    x, y, z = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2]
    keep = (
        (x >= x_range[0]) & (x < x_range[1])
        & (y >= y_range[0]) & (y < y_range[1])
        & (z >= z_range[0]) & (z <= z_range[1])
    )
    x, y, z = x[keep], y[keep], z[keep]

    rows = np.clip(((x_range[1] - x) / resolution).astype(np.int64), 0, h - 1)
    cols = np.clip(((y_range[1] - y) / resolution).astype(np.int64), 0, w - 1)
    flat = rows * w + cols

    count = np.zeros(h * w, dtype=np.float64)
    np.add.at(count, flat, 1.0)
    zsum = np.zeros(h * w, dtype=np.float64)
    np.add.at(zsum, flat, z)
    zmax = np.full(h * w, -np.inf)
    np.maximum.at(zmax, flat, z)
    zmin = np.full(h * w, np.inf)
    np.minimum.at(zmin, flat, z)

    occupied = count > 0
    mean = np.where(occupied, zsum / np.maximum(count, 1.0), 0.0)
    zmax = np.where(occupied, zmax, 0.0)
    zmin = np.where(occupied, zmin, 0.0)
    rng = np.where(occupied, zmax - zmin, 0.0)

    shape = (h, w)
    return BevTensor(
        density=count.reshape(shape),
        max_height=zmax.reshape(shape),
        min_height=zmin.reshape(shape),
        mean_height=mean.reshape(shape),
        height_range=rng.reshape(shape),
        occupied=occupied.reshape(shape),
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )


def bev_tensor_from_lidar(
    lidar: LidarFrame,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-3.0, 5.0),
    resolution: float = 0.25,
) -> BevTensor:
    points_ego = lidar_points_to_ego(load_lidar_points(lidar.pointcloud_path), lidar)
    return compute_bev_height_channels(points_ego, x_range, y_range, z_range, resolution)


def _colorize(channel: np.ndarray, occupied: np.ndarray, vmin: float, vmax: float, cmap: int) -> np.ndarray:
    norm = np.clip((channel - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    color[~occupied] = (20, 20, 20)
    return color


def _panel(image: np.ndarray, label: str, tile: int) -> np.ndarray:
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (tile, int(round(h * tile / w))), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(scaled, (0, 0), (scaled.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(scaled, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return scaled


def render_bev_channels(
    frame: SurroundFrame,
    tensor: BevTensor,
    z_range: tuple[float, float] = (-3.0, 5.0),
    tile: int = 300,
) -> np.ndarray:
    """Tile the geometric channels into one inspection sheet."""
    occ = tensor.occupied
    density_norm = np.log1p(tensor.density)
    density_vmax = float(density_norm.max()) if occ.any() else 1.0

    panels = [
        _panel(_colorize(density_norm, occ, 0.0, density_vmax, cv2.COLORMAP_MAGMA), "density (log)", tile),
        _panel(_colorize(tensor.max_height, occ, z_range[0], z_range[1], cv2.COLORMAP_VIRIDIS), "max height", tile),
        _panel(_colorize(tensor.min_height, occ, z_range[0], z_range[1], cv2.COLORMAP_VIRIDIS), "min height", tile),
        _panel(_colorize(tensor.mean_height, occ, z_range[0], z_range[1], cv2.COLORMAP_VIRIDIS), "mean height", tile),
        _panel(_colorize(tensor.height_range, occ, 0.0, 4.0, cv2.COLORMAP_INFERNO), "height range", tile),
    ]

    info = np.full((panels[0].shape[0], tile, 3), 24, dtype=np.uint8)
    occ_cells = int(occ.sum())
    lines = [
        f"{frame.scene_name}",
        f"sample {frame.sample_index}",
        f"occupied cells: {occ_cells}",
        f"grid {tensor.density.shape[0]}x{tensor.density.shape[1]}",
        f"res {tensor.resolution:.2f} m/cell",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info, text, (8, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 230, 255), 1, cv2.LINE_AA)
    panels.append(info)

    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    return np.vstack([top, bottom])


def save_bev_tensor_sequence(
    dataroot: str | Path | None = None,
    scene_index: int = 0,
    scene_name: str | None = None,
    start_sample_index: int = 0,
    max_frames: int | None = 40,
    output_path: str | Path = "outputs/nuscenes_bev_tensor_scene0_40f.mp4",
    fps: float = 2.0,
    resolution: float = 0.25,
    z_range: tuple[float, float] = (-3.0, 5.0),
) -> Path:
    loader = NuScenesSceneLoader(dataroot=dataroot)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rendered = 0
    try:
        for frame, lidar in loader.iter_scene_frames(
            scene_index=scene_index,
            start_sample_index=start_sample_index,
            max_frames=max_frames,
            scene_name=scene_name,
            include_lidar=True,
        ):
            if lidar is None:
                raise RuntimeError("LiDAR frame was not loaded")
            tensor = bev_tensor_from_lidar(lidar, resolution=resolution, z_range=z_range)
            image = render_bev_channels(frame, tensor, z_range=z_range)
            rendered += 1
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not open video writer: {output}")
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()

    if rendered == 0:
        raise RuntimeError("No BEV tensor frames were rendered")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 2.5D LiDAR BEV height-channel inspection videos.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to render.")
    parser.add_argument("--scene-name", default=None, help="Scene name to render, e.g. scene-0001.")
    parser.add_argument("--start-sample-index", type=int, default=0, help="First key sample index within the scene.")
    parser.add_argument("--frames", type=int, default=40, help="Maximum number of key samples to render.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS.")
    parser.add_argument("--resolution", type=float, default=0.25, help="BEV grid meters per cell.")
    parser.add_argument("--output", default="outputs/nuscenes_bev_tensor_scene0_40f.mp4", help="Output video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_bev_tensor_sequence(
        dataroot=args.dataroot,
        scene_index=args.scene_index,
        scene_name=args.scene_name,
        start_sample_index=args.start_sample_index,
        max_frames=args.frames,
        output_path=args.output,
        fps=args.fps,
        resolution=args.resolution,
    )
    print(output)


if __name__ == "__main__":
    main()
