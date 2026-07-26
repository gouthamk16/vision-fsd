from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from fsd.bev import render_lidar_bev
from fsd.contact_sheet import render_contact_sheet
from fsd.data import NuScenesSceneLoader
from fsd.lidar_projection import render_lidar_projection_sheet
from fsd.object_detection import NuScenesAnnotationLoader, PredictionLoader
from fsd.lss import LSSInference, overlay_lss_on_lidar_bev, render_lss_bev
from fsd.motion_planning.render import render_planning_camera_result, render_planning_result
from fsd.motion_planning.runner import OfflinePlanningRuntime, OfflinePlanningRuntimeConfig
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.occupancy import TemporalOccupancyMapper, render_occupancy_bev
from fsd.bev_tensor import bev_tensor_from_lidar, render_bev_channels
from fsd.fusion_detect import FrustumFusionDetector, render_fusion_bev, render_fusion_cameras
from fsd.world_model import WorldModelBuilder, WorldModelConfig, render_world_model_bev


def _default_output_path(view: str, scenes_label: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"nuscenes_{view}_{scenes_label}_{timestamp}.mp4"


def _render_view(
    view,
    frame,
    lidar,
    tile_width,
    max_depth,
    point_radius,
    bev_resolution,
    bev_scale,
    lss_inference=None,
    lss_threshold=0.4,
    map_renderer=None,
    occupancy_mapper=None,
    planner_runtime=None,
    detector=None,
    world_builder=None,
):
    if view == "planner_bev":
        if planner_runtime is None:
            raise RuntimeError("Planner BEV view requested but no planner runtime was created")
        if lidar is None:
            raise RuntimeError("Planner BEV view requested but no LiDAR frame was loaded")
        result = planner_runtime.step(frame, lidar)
        return render_planning_result(result, scale=bev_scale)
    if view == "planner_camera":
        if planner_runtime is None:
            raise RuntimeError("Planner camera view requested but no planner runtime was created")
        if lidar is None:
            raise RuntimeError("Planner camera view requested but no LiDAR frame was loaded")
        result = planner_runtime.step(frame, lidar)
        return render_planning_camera_result(frame, lidar, result, tile_width=tile_width)
    if view == "world_bev":
        if world_builder is None:
            raise RuntimeError("World model view requested but no world builder was created")
        if lidar is None:
            raise RuntimeError("World model view requested but no LiDAR frame was loaded")
        return render_world_model_bev(world_builder.step(frame, lidar), scale=bev_scale)
    if view in {"objects_bev", "objects_cameras"}:
        if detector is None:
            raise RuntimeError("Object detection view requested but no detector was created")
        boxes = detector.detect(frame, lidar)
        if view == "objects_bev":
            return render_fusion_bev(frame, lidar, boxes, resolution=bev_resolution, scale=bev_scale)
        return render_fusion_cameras(frame, boxes, tile_width=tile_width)
    if view == "occupancy_bev":
        if occupancy_mapper is None:
            raise RuntimeError("Occupancy view requested but no occupancy mapper was created")
        prob = occupancy_mapper.step(lidar)
        return render_occupancy_bev(frame, prob, resolution=bev_resolution, scale=bev_scale)
    if view == "height_bev":
        tensor = bev_tensor_from_lidar(lidar, resolution=bev_resolution)
        return render_bev_channels(frame, tensor)
    if view == "lidar":
        return render_lidar_projection_sheet(frame, lidar, tile_width=tile_width, max_depth=max_depth, point_radius=point_radius)
    if view == "bev":
        return render_lidar_bev(frame, lidar, resolution=bev_resolution, scale=bev_scale)
    if view == "lss_bev":
        if lss_inference is None:
            raise RuntimeError("LSS BEV view requested but --lss-weights was not provided")
        prob = lss_inference.infer(frame)
        map_bg = None
        if map_renderer is not None:
            # 200x200 LSS native, render_lss_bev upsamples 4x -> 800x800.
            map_bg = map_renderer.render(frame, output_hw=(800, 800))
        return render_lss_bev(frame, prob, threshold=lss_threshold, map_background=map_bg)
    if view == "lss_lidar_bev":
        if lss_inference is None:
            raise RuntimeError("LSS+LiDAR BEV view requested but --lss-weights was not provided")
        prob = lss_inference.infer(frame)
        lidar_bev = render_lidar_bev(frame, lidar, resolution=bev_resolution, scale=bev_scale)
        return overlay_lss_on_lidar_bev(
            lidar_bev,
            prob,
            lidar_resolution=bev_resolution,
            lidar_scale=bev_scale,
            threshold=lss_threshold,
        )
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
    min_lidar_points: int = 1,
    predictions_path: str | Path | None = None,
    score_threshold: float = 0.1,
    lss_weights: str | Path | None = None,
    lss_threshold: float = 0.4,
    lss_device: str | None = None,
    use_map: bool = True,
    yolo_weights: str | Path = "yolo26n.pt",
    detector_device: str | None = None,
    wait_ms: int = 1,
) -> list[Path] | None:
    valid_views = {
        "cameras",
        "lidar",
        "bev",
        "lss_bev",
        "lss_lidar_bev",
        "occupancy_bev",
        "planner_bev",
        "planner_camera",
        "height_bev",
        "objects_bev",
        "objects_cameras",
        "world_bev",
        "all",
    }
    if view not in valid_views:
        raise ValueError(f"view must be one of {sorted(valid_views)}")
    if sequence not in {"keyframes", "sweeps"}:
        raise ValueError("sequence must be 'keyframes' or 'sweeps'")
    if sequence == "sweeps" and view != "cameras":
        raise ValueError("camera sweeps currently support --view cameras only")
    if mode not in {"save", "stream"}:
        raise ValueError("mode must be 'save' or 'stream'")

    if scene_indices is None:
        scene_indices = [0]

    loader = NuScenesSceneLoader(dataroot=dataroot)
    all_views = [
        "cameras",
        "lidar",
        "bev",
        "lss_bev",
        "lss_lidar_bev",
        "occupancy_bev",
        "planner_bev",
        "planner_camera",
        "height_bev",
        "objects_bev",
        "objects_cameras",
        "world_bev",
    ]
    views = all_views if view == "all" else [view]
    include_lidar = any(
        v in {
            "lidar", "bev", "lss_lidar_bev",
            "occupancy_bev", "planner_bev", "planner_camera", "height_bev", "objects_bev", "objects_cameras",
            "world_bev",
        }
        for v in views
    )
    needs_gt = "world_bev" in views
    needs_lss = any(v in {"lss_bev", "lss_lidar_bev"} for v in views)
    needs_occupancy = "occupancy_bev" in views
    needs_planner = any(v in {"planner_bev", "planner_camera"} for v in views)
    needs_detector = any(v in {"objects_bev", "objects_cameras"} for v in views)
    needs_world = "world_bev" in views
    annotation_loader = NuScenesAnnotationLoader(loader) if needs_gt else None
    prediction_loader = PredictionLoader(predictions_path) if predictions_path else None
    lss_inference = None
    if needs_lss:
        if not lss_weights:
            raise ValueError("--lss-weights is required for lss_bev and lss_lidar_bev views")
        lss_inference = LSSInference(weights_path=lss_weights, device=lss_device)
    map_renderer = None
    if needs_lss and use_map:
        try:
            map_renderer = NuScenesMapRenderer(scene_loader=loader)
        except FileNotFoundError:
            map_renderer = None
    detector = FrustumFusionDetector(weights_path=yolo_weights, device=detector_device) if needs_detector else None
    planner_runtime = (
        OfflinePlanningRuntime(OfflinePlanningRuntimeConfig(resolution=bev_resolution))
        if needs_planner
        else None
    )
    world_builder = (
        WorldModelBuilder(
            WorldModelConfig(
                resolution=bev_resolution,
                min_lidar_points=min_lidar_points,
                score_threshold=score_threshold,
            ),
            annotation_loader=annotation_loader,
            prediction_loader=prediction_loader,
        )
        if needs_world
        else None
    )

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

            # Occupancy is stateful — fresh rolling grid per scene.
            occupancy_mapper = TemporalOccupancyMapper(resolution=bev_resolution) if needs_occupancy else None
            if planner_runtime is not None:
                planner_runtime.reset()
            if world_builder is not None:
                world_builder.reset()

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
                if annotation_loader is not None:
                    scene = loader._select_scene(scene_index=scene_idx, scene_name=name)
                    samples = loader._sample_sequence(scene, start, max_frames)
                    annotation_loader.prefetch_sample_annotations({sample["token"] for _, sample in samples})
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
                    image = _render_view(
                        v,
                        frame,
                        lidar,
                        tile_width,
                        max_depth,
                        point_radius,
                        bev_resolution,
                        bev_scale,
                        lss_inference=lss_inference,
                        lss_threshold=lss_threshold,
                        map_renderer=map_renderer,
                        occupancy_mapper=occupancy_mapper,
                        planner_runtime=planner_runtime,
                        detector=detector,
                        world_builder=world_builder,
                    )
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
    parser.add_argument(
        "--view",
        choices=(
            "cameras",
            "lidar",
            "bev",
            "lss_bev",
            "lss_lidar_bev",
            "occupancy_bev",
            "planner_bev",
            "planner_camera",
            "height_bev",
            "objects_bev",
            "objects_cameras",
            "world_bev",
            "all",
        ),
        default="lidar",
        help="Visualization view.",
    )
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
    parser.add_argument("--min-lidar-points", type=int, default=1, help="Minimum LiDAR points required per GT box.")
    parser.add_argument("--predictions", default=None, help="Path to CenterPoint/MMDetection3D prediction JSON.")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Minimum prediction confidence to draw.")
    parser.add_argument("--lss-weights", default=None, help="Path to pretrained LSS BEV vehicle-seg checkpoint.")
    parser.add_argument("--lss-threshold", type=float, default=0.4, help="LSS vehicle-seg probability threshold for overlays.")
    parser.add_argument("--lss-device", default=None, help="Device for LSS inference, e.g. 'cuda', 'cuda:0', 'cpu'.")
    parser.add_argument("--no-map", action="store_true", help="Disable nuScenes HD-map background for LSS views.")
    parser.add_argument("--yolo-weights", default="yolo26n.pt", help="YOLO weights for the fusion object detector.")
    parser.add_argument("--detector-device", default=None, help="Device for the object detector, e.g. 'cuda', 'cpu'.")
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
        min_lidar_points=args.min_lidar_points,
        predictions_path=args.predictions,
        score_threshold=args.score_threshold,
        lss_weights=args.lss_weights,
        lss_threshold=args.lss_threshold,
        lss_device=args.lss_device,
        use_map=not args.no_map,
        yolo_weights=args.yolo_weights,
        detector_device=args.detector_device,
        wait_ms=args.wait_ms,
    )
    if outputs:
        for path in outputs:
            print(path)


if __name__ == "__main__":
    main()
