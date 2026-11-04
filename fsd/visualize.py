from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from fsd.bev import BevGrid, draw_ego, lidar_canvas, render_lidar_bev
from fsd.bev_tensor import compute_bev_height_channels, render_bev_channels
from fsd.contact_sheet import render_contact_sheet
from fsd.data import LidarFrame, NuScenesSceneLoader, SurroundFrame
from fsd.draw import with_header
from fsd.fusion_detect import FrustumFusionDetector, render_fusion_bev, render_fusion_cameras
from fsd.lidar_projection import lidar_points_ego, render_lidar_projection_sheet
from fsd.lss import LSSInference, overlay_lss, render_lss_bev
from fsd.nmp.config import NmpConfig
from fsd.nmp.planner import NmpPlanner
from fsd.nmp.render import render_planner_bev
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.object_detection import NuScenesAnnotationLoader, PredictionLoader
from fsd.occupancy import TemporalOccupancyMapper, render_occupancy_bev
from fsd.world_model import WorldModelBuilder, render_world_model_bev

LSS_VIEWS = {"lss_bev", "lss_lidar_bev"}
CAMERA_ONLY_VIEWS = {"cameras", "lss_bev"}
MAP_VIEWS = {"bev", "lss_bev", "objects_bev", "world_bev", "planner_bev"}


@dataclass
class _Context:
    """Components built once per run, plus render settings."""

    raster: BevGrid
    view: BevGrid
    tile_width: int
    max_depth: float
    point_radius: int
    lss_threshold: float
    lidar_sweeps: int
    mapper: TemporalOccupancyMapper | None = None
    world: WorldModelBuilder | None = None
    detector: FrustumFusionDetector | None = None
    lss: LSSInference | None = None
    map_renderer: NuScenesMapRenderer | None = None
    planner: NmpPlanner | None = None
    # The planner's map input is needed even when the map background is off (--no-map).
    planner_map: NuScenesMapRenderer | None = None


class _Products:
    """Per-frame data shared by every view, each computed on first use."""

    def __init__(self, frame: SurroundFrame, lidar: LidarFrame | None, ctx: _Context):
        self.frame, self.lidar, self.ctx = frame, lidar, ctx

    @cached_property
    def points(self) -> np.ndarray:
        # The planner may have loaded more history than the views should draw.
        return lidar_points_ego(LidarFrame(self.lidar.sweeps[: self.ctx.lidar_sweeps]))

    @cached_property
    def occupancy(self) -> np.ndarray:
        return self.ctx.mapper.step(self.frame, self.lidar, self.points)

    @cached_property
    def detections(self):
        return self.ctx.detector.detect(self.frame, self.lidar, self.points)

    @cached_property
    def lss(self) -> np.ndarray:
        return self.ctx.lss.infer(self.frame)

    @cached_property
    def dark_map(self) -> np.ndarray | None:
        return self.ctx.map_renderer.render(self.frame, self.ctx.view, dark=True) if self.ctx.map_renderer else None


def _lss_lidar(p: _Products, c: _Context) -> np.ndarray:
    image = overlay_lss(lidar_canvas(p.points, c.view), p.lss, c.view, c.lss_threshold)
    draw_ego(image, c.view)
    return with_header(image, f"{p.frame.scene_name} | sample {p.frame.sample_index} | LSS (cyan) over LiDAR BEV")


RENDERERS: dict[str, Callable[[_Products, _Context], np.ndarray]] = {
    "cameras": lambda p, c: render_contact_sheet(p.frame, c.tile_width),
    "lidar": lambda p, c: render_lidar_projection_sheet(p.frame, p.lidar, p.points, c.tile_width, c.max_depth, c.point_radius),
    "bev": lambda p, c: render_lidar_bev(p.frame, p.points, c.view, p.dark_map),
    "lss_bev": lambda p, c: render_lss_bev(p.frame, p.lss, c.view, c.map_renderer, c.lss_threshold),
    "lss_lidar_bev": _lss_lidar,
    "occupancy_bev": lambda p, c: render_occupancy_bev(p.frame, p.occupancy, c.raster, c.view),
    "height_bev": lambda p, c: render_bev_channels(p.frame, compute_bev_height_channels(p.points, c.raster), c.view.with_resolution(c.view.resolution * 2)),
    "objects_bev": lambda p, c: render_fusion_bev(p.frame, p.points, p.detections, c.view, p.dark_map),
    "objects_cameras": lambda p, c: render_fusion_cameras(p.frame, p.detections, c.tile_width),
    "world_bev": lambda p, c: render_world_model_bev(c.world.step(p.frame, p.occupancy), c.raster, c.view, c.map_renderer),
    "planner_bev": lambda p, c: render_planner_bev(p.frame, p.points, c.planner.plan(p.frame, p.lidar, c.planner_map), c.planner.cfg.grid, c.view, p.dark_map),
}
VIEWS = tuple(RENDERERS)


def _select_views(view: str, lss_weights: str | None, nmp_weights: str | None) -> list[str]:
    missing = {v: "--lss-weights" for v in LSS_VIEWS if not lss_weights}
    if not nmp_weights:
        missing["planner_bev"] = "--nmp-weights"
    if view != "all":
        if view in missing:
            raise ValueError(f"{missing[view]} is required for {view}")
        return [view]
    return [v for v in VIEWS if v not in missing]


def _build_context(views: list[str], loader: NuScenesSceneLoader, opts: argparse.Namespace) -> _Context:
    raster = BevGrid(resolution=opts.bev_resolution)
    ctx = _Context(raster, raster.with_resolution(1.0 / opts.bev_ppm), opts.tile_width, opts.max_depth, opts.point_radius, opts.lss_threshold, opts.lidar_sweeps)
    if {"occupancy_bev", "world_bev"} & set(views):
        ctx.mapper = TemporalOccupancyMapper(raster)
    if "world_bev" in views:
        predictions = PredictionLoader(opts.predictions) if opts.predictions else None
        ctx.world = WorldModelBuilder(NuScenesAnnotationLoader(loader), predictions, opts.min_lidar_points, opts.score_threshold)
    if {"objects_bev", "objects_cameras"} & set(views):
        ctx.detector = FrustumFusionDetector(opts.yolo_weights, opts.detector_device)
    if LSS_VIEWS & set(views):
        ctx.lss = LSSInference(opts.lss_weights, opts.lss_device)
    if "planner_bev" in views:
        ctx.planner = NmpPlanner(opts.nmp_weights)
        ctx.planner_map = NuScenesMapRenderer(loader.dataroot)
    if not opts.no_map and MAP_VIEWS & set(views):
        try:
            ctx.map_renderer = NuScenesMapRenderer(loader.dataroot)
        except FileNotFoundError as exc:
            print(f"Map disabled: {exc}")
    return ctx


def _frames(loader: NuScenesSceneLoader, opts: argparse.Namespace, needs_lidar: bool, sweeps: int) -> Iterator[tuple[SurroundFrame, LidarFrame | None]]:
    for i, scene_index in enumerate([0] if opts.scene_name else opts.scenes):
        start = opts.start_sample_index if i == 0 else 0
        if opts.sequence == "sweeps":
            for frame in loader.iter_camera_sweep_frames(scene_index, start, opts.frames, opts.scene_name):
                yield frame, None
        else:
            yield from loader.iter_scene_frames(scene_index, start, opts.frames, opts.scene_name, needs_lidar, sweeps)


def _output_paths(output: str | None, views: list[str]) -> dict[str, Path]:
    if output:
        base = Path(output)
        return {views[0]: base} if len(views) == 1 else {v: base.with_name(f"{base.stem}_{v}{base.suffix}") for v in views}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return {v: Path("outputs") / f"nuscenes_{v}_{stamp}.mp4" for v in views}


class _Sink:
    """Writes each view to its own MP4, or shows it in a window."""

    def __init__(self, opts: argparse.Namespace, views: list[str]):
        self.stream, self.fps, self.wait_ms = opts.stream, opts.fps, opts.wait_ms
        self.paths = _output_paths(opts.output, views)
        self.writers: dict[str, cv2.VideoWriter] = {}

    def show(self, view: str, image: np.ndarray) -> None:
        if self.stream:
            cv2.imshow(f"nuScenes {view}", image)
            return
        if view not in self.writers:
            self.paths[view].parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(self.paths[view]), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, image.shape[1::-1])
            if not writer.isOpened():
                raise OSError(f"Could not open video writer: {self.paths[view]}")
            self.writers[view] = writer
        self.writers[view].write(image)

    def quit_requested(self) -> bool:
        return self.stream and (cv2.waitKey(self.wait_ms) & 0xFF) == ord("q")

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()
        if self.stream:
            cv2.destroyAllWindows()


def run_visualizer(opts: argparse.Namespace) -> list[Path]:
    """Render the requested views; returns the MP4s written (none when streaming)."""
    if opts.sequence == "sweeps" and opts.view != "cameras":
        raise ValueError("--sequence sweeps supports --view cameras only")
    views = _select_views(opts.view, opts.lss_weights, opts.nmp_weights)
    loader = NuScenesSceneLoader(dataroot=opts.dataroot)
    ctx = _build_context(views, loader, opts)
    sink = _Sink(opts, views)
    rendered = 0
    try:
        sweeps = max(opts.lidar_sweeps, NmpConfig().history_sweeps) if ctx.planner else opts.lidar_sweeps
        for frame, lidar in _frames(loader, opts, any(v not in CAMERA_ONLY_VIEWS for v in views), sweeps):
            products = _Products(frame, lidar, ctx)
            for view in views:
                sink.show(view, RENDERERS[view](products, ctx))
            rendered += 1
            if sink.quit_requested():
                break
    finally:
        sink.close()
    if rendered == 0:
        raise RuntimeError("No frames were rendered")
    return [sink.paths[v] for v in sink.writers]


def _scenes_arg(value: str) -> list[int]:
    lo, _, hi = value.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _frames_arg(value: str) -> int | None:
    if value.lower() == "all":
        return None
    if int(value) < 0:
        raise argparse.ArgumentTypeError("--frames must be a non-negative integer or 'all'")
    return int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render nuScenes visualizations to MP4 or a window.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or C:/nuscenes.")
    parser.add_argument("--scenes", type=_scenes_arg, default=[0], help="Available-scene index or range, e.g. '5' or '0-9'.")
    parser.add_argument("--scene-name", default=None, help="Named scene, e.g. scene-0061. Overrides --scenes.")
    parser.add_argument("--start-sample-index", type=int, default=0, help="First keyframe or sweep index (first scene only).")
    parser.add_argument("--frames", type=_frames_arg, default=20, help="Max frames per scene, or 'all'.")
    parser.add_argument("--view", choices=VIEWS + ("all",), default="lidar")
    parser.add_argument("--sequence", choices=("keyframes", "sweeps"), default="keyframes", help="Key samples, or every camera sweep (cameras view only).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save", action="store_true", help="Save one MP4 per view (the default).")
    mode.add_argument("--stream", action="store_true", help="Show in a window instead of saving MP4s.")
    parser.add_argument("--output", default=None, help="MP4 path; with several views, used as a base suffixed per view.")
    parser.add_argument("--tile-width", type=int, default=480, help="Camera tile width in pixels.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Depth (m) at the end of the LiDAR colour scale.")
    parser.add_argument("--point-radius", type=int, default=1, help="Projected LiDAR point radius in tile pixels.")
    parser.add_argument("--lidar-sweeps", type=int, default=5, help="LiDAR sweeps per keyframe (20 Hz, ego-motion compensated).")
    parser.add_argument("--bev-resolution", type=float, default=0.25, help="Occupancy / height raster resolution, m per cell.")
    parser.add_argument("--bev-ppm", type=float, default=10.0, help="BEV view resolution, pixels per metre.")
    parser.add_argument("--min-lidar-points", type=int, default=1, help="Minimum LiDAR points for a GT box to be drawn.")
    parser.add_argument("--predictions", default=None, help="CenterPoint / nuScenes-format prediction JSON for world_bev.")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Minimum prediction score.")
    parser.add_argument("--lss-weights", default=None, help="Pretrained LSS vehicle-seg checkpoint (required for LSS views).")
    parser.add_argument("--lss-threshold", type=float, default=0.4, help="LSS probability threshold for overlays.")
    parser.add_argument("--lss-device", default=None, help="Device for LSS, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--nmp-weights", default=None, help="Neural motion planner checkpoint (required for planner_bev).")
    parser.add_argument("--no-map", action="store_true", help="Disable the HD-map background.")
    parser.add_argument("--yolo-weights", default="yolo26n.pt", help="YOLO weights for the fusion detector.")
    parser.add_argument("--detector-device", default=None, help="Device for the fusion detector.")
    parser.add_argument("--wait-ms", type=int, default=1, help="OpenCV wait per frame with --stream.")
    return parser.parse_args(argv)


def main() -> None:
    for path in run_visualizer(parse_args()):
        print(path)


if __name__ == "__main__":
    main()
