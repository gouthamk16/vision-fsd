from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from fsd.bev import render_lidar_bev
from fsd.contact_sheet import render_contact_sheet
from fsd.data import NuScenesSceneLoader
from fsd.lidar_projection import render_lidar_projection_sheet


def _default_output_path(view: str, scenes_label: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"nuscenes_{view}_{scenes_label}_{timestamp}.mp4"


def _render_view(view, frame, lidar, tile_width, max_depth, point_radius, bev_resolution, bev_scale):
    if view == "lidar":
        return render_lidar_projection_sheet(frame, lidar, tile_width=tile_width, max_depth=max_depth, point_radius=point_radius)
    if view == "bev":
        return render_lidar_bev(frame, lidar, resolution=bev_resolution, scale=bev_scale)
    return render_contact_sheet(frame, tile_width=tile_width)


def _scenes_arg(value: str) -> list[int]:
    if "-" in value:
        lo, _, hi = value.partition("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(value)]


def _frames_arg(value: str) -> int | None:
    if value.lower() == "all":
        return None
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("--frames must be a positive integer or 'all'")
    return n


def run_visualizer(
    dataroot: str | Path | None = None,
    scene_indices: list[int] | None = None,
    scene_name: str | None = None,
    start_sample_index: int = 0,
    max_frames: int | None = 20,
    view: str = "lidar",
    sequence: str = "keyframes",
    mode: str = "save",
    output_path: str | Path | None = None,
    tile_width: int = 360,
    fps: float = 2.0,
    max_depth: float = 80.0,
    point_radius: int = 1,
    bev_resolution: float = 0.25,
    bev_scale: int = 2,
    wait_ms: int = 1,
) -> list[Path] | None:
    if view not in {"cameras", "lidar", "bev", "all"}:
        raise ValueError("view must be 'cameras', 'lidar', 'bev', or 'all'")
    if sequence not in {"keyframes", "sweeps"}:
        raise ValueError("sequence must be 'keyframes' or 'sweeps'")
    if sequence == "sweeps" and view != "cameras":
        raise ValueError("camera sweeps currently support --view cameras only")
    if mode not in {"save", "stream"}:
        raise ValueError("mode must be 'save' or 'stream'")

    if scene_indices is None:
        scene_indices = [0]

    loader = NuScenesSceneLoader(dataroot=dataroot)
    views = ["cameras", "lidar", "bev"] if view == "all" else [view]
    include_lidar = any(v in {"lidar", "bev"} for v in views)

    scenes_label = f"scene{scene_indices[0]}" if len(scene_indices) == 1 else f"scenes{scene_indices[0]}-{scene_indices[-1]}"
    outputs: dict[str, Path] = {}
    if mode == "save":
        if view == "all":
            if output_path:
                base = Path(output_path)
                outputs = {v: base.parent / f"{base.stem}_{v}{base.suffix}" for v in views}
            else:
                outputs = {v: _default_output_path(v, scenes_label) for v in views}
        else:
            out = Path(output_path) if output_path else _default_output_path(view, scenes_label)
            outputs = {view: out}

    writers: dict[str, cv2.VideoWriter] = {}
    rendered_count = 0
    aborted = False

    try:
        for scene_i, scene_idx in enumerate(scene_indices):
            if aborted:
                break

            name = scene_name if len(scene_indices) == 1 else None
            start = start_sample_index if scene_i == 0 else 0

            if sequence == "sweeps":
                frame_iter = (
                    (frame, None)
                    for frame in loader.iter_camera_sweep_frames(
                        scene_index=scene_idx,
                        start_frame_index=start,
                        max_frames=max_frames,
                        scene_name=name,
                    )
                )
            else:
                frame_iter = loader.iter_scene_frames(
                    scene_index=scene_idx,
                    start_sample_index=start,
                    max_frames=max_frames,
                    scene_name=name,
                    include_lidar=include_lidar,
                )

            for frame, lidar in frame_iter:
                if include_lidar and lidar is None:
                    raise RuntimeError("LiDAR view requested but no LiDAR frame was loaded")

                rendered_count += 1

                for v in views:
                    image = _render_view(v, frame, lidar, tile_width, max_depth, point_radius, bev_resolution, bev_scale)
                    cv2.putText(
                        image,
                        f"frame {rendered_count}",
                        (image.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )

                    if mode == "save":
                        if v not in writers:
                            outputs[v].parent.mkdir(parents=True, exist_ok=True)
                            h, w = image.shape[:2]
                            writer = cv2.VideoWriter(str(outputs[v]), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
                            if not writer.isOpened():
                                raise OSError(f"Could not open video writer: {outputs[v]}")
                            writers[v] = writer
                        writers[v].write(image)
                    else:
                        cv2.imshow(f"nuScenes {v}", image)

                if mode == "stream":
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q"):
                        aborted = True
                        break
    finally:
        for writer in writers.values():
            writer.release()
        if mode == "stream":
            cv2.destroyAllWindows()

    if rendered_count == 0:
        raise RuntimeError("No frames were rendered")

    if mode == "save":
        return list(outputs.values())
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a nuScenes 360 visualization sequence.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--scenes", type=_scenes_arg, default=[0], help="Scene index or range, e.g. '5' or '0-9'.")
    parser.add_argument("--scene-name", default=None, help="Named scene to render, e.g. scene-0001. Overrides --scenes.")
    parser.add_argument("--start-sample-index", type=int, default=0, help="First key sample or sweep index (first scene only).")
    parser.add_argument("--frames", type=_frames_arg, default=20, help="Max frames per scene. Pass 'all' for every frame in each scene.")
    parser.add_argument("--view", choices=("cameras", "lidar", "bev", "all"), default="lidar", help="Visualization view.")
    parser.add_argument(
        "--sequence",
        choices=("keyframes", "sweeps"),
        default="keyframes",
        help="Use sparse key samples or grouped camera sweeps.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save", action="store_true", help="Save the rendered sequence as an MP4.")
    mode.add_argument("--stream", action="store_true", help="Display the rendered sequence in an OpenCV window.")
    parser.add_argument("--output", default=None, help="Output path for --save mode. With --view all, used as base (suffixed per view).")
    parser.add_argument("--tile-width", type=int, default=360, help="Width of each camera tile in pixels.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Depth in meters used for LiDAR color scaling.")
    parser.add_argument("--point-radius", type=int, default=1, help="Projected LiDAR point radius in pixels.")
    parser.add_argument("--bev-resolution", type=float, default=0.25, help="BEV grid meters per cell.")
    parser.add_argument("--bev-scale", type=int, default=2, help="Nearest-neighbor scale for BEV output.")
    parser.add_argument("--wait-ms", type=int, default=1, help="OpenCV wait time for --stream mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "stream" if args.stream else "save"
    scene_indices = None if args.scene_name else args.scenes
    outputs = run_visualizer(
        dataroot=args.dataroot,
        scene_indices=scene_indices,
        scene_name=args.scene_name,
        start_sample_index=args.start_sample_index,
        max_frames=args.frames,
        view=args.view,
        sequence=args.sequence,
        mode=mode,
        output_path=args.output,
        tile_width=args.tile_width,
        fps=args.fps,
        max_depth=args.max_depth,
        point_radius=args.point_radius,
        bev_resolution=args.bev_resolution,
        bev_scale=args.bev_scale,
        wait_ms=args.wait_ms,
    )
    if outputs:
        for path in outputs:
            print(path)


if __name__ == "__main__":
    main()
