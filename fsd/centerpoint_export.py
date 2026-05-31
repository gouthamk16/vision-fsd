"""Run pretrained CenterPoint (mmdet3d) on nuScenes LiDAR and export ego-frame
3D boxes as prediction JSON for the visualizer's pred_bev / compare_bev views.

Runs ONLY in the isolated mmdet3d environment (.venv-mmdet3d): torch 2.1 +
mmcv/mmdet/mmdet3d + spconv + nuscenes-devkit. The main .venv cannot import it.

nuScenes CenterPoint is trained on 10 accumulated LiDAR sweeps with a per-point
time channel. `inference_detector` on a single .pcd.bin only sees one sweep, so
we aggregate the sweeps ourselves (devkit transforms) into the current LiDAR
frame, write a 5-dim (x,y,z,intensity,time) cloud, and strip the pipeline's
LoadPointsFromMultiSweeps so it isn't re-padded.

Output (consumed by fsd.object_detection.PredictionLoader):
    {"samples": {sample_token: [{center_ego, yaw, size[w,l,h], detection_name,
                                 detection_score}, ...]}}
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import reduce
from pathlib import Path

import numpy as np

from fsd.data import NuScenesSceneLoader
from fsd.lidar_projection import quaternion_to_rotation_matrix


# nuScenes class order used by the CenterPoint nuScenes configs.
NUSC_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def _yaw_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _aggregate_sweeps(nusc, sample, nsweeps: int, min_distance: float = 1.0) -> tuple[np.ndarray, dict]:
    """Accumulate nsweeps LiDAR sweeps into the current LIDAR_TOP frame.

    Returns (Nx5 points [x,y,z,intensity,time], ref calibrated_sensor record).
    Mirrors nuScenes/mmdet3d LoadPointsFromMultiSweeps.
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    ref_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ref_cs = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])
    ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    ref_time = 1e-6 * ref_sd["timestamp"]

    ref_from_car = transform_matrix(ref_cs["translation"], Quaternion(ref_cs["rotation"]), inverse=True)
    car_from_global = transform_matrix(ref_pose["translation"], Quaternion(ref_pose["rotation"]), inverse=True)

    points = np.zeros((5, 0), dtype=np.float64)
    sd = ref_sd
    for _ in range(nsweeps):
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"]))
        pc.remove_close(min_distance)

        cur_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        cur_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        global_from_car = transform_matrix(cur_pose["translation"], Quaternion(cur_pose["rotation"]), inverse=False)
        car_from_current = transform_matrix(cur_cs["translation"], Quaternion(cur_cs["rotation"]), inverse=False)
        pc.transform(reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]))

        times = (ref_time - 1e-6 * sd["timestamp"]) * np.ones((1, pc.nbr_points()))
        points = np.concatenate([points, np.concatenate([pc.points, times], axis=0)], axis=1)

        if sd["prev"] == "":
            break
        sd = nusc.get("sample_data", sd["prev"])

    return points.T.astype(np.float32), ref_cs


def _boxes_to_ego_records(pred, ref_cs: dict, score_threshold: float) -> list[dict]:
    """Convert LiDAR-sensor-frame CenterPoint boxes to ego-frame records."""
    rot_cs = quaternion_to_rotation_matrix(ref_cs["rotation"])
    trans_cs = np.asarray(ref_cs["translation"], dtype=np.float64)

    centers = pred.bboxes_3d.gravity_center.cpu().numpy()
    dims = pred.bboxes_3d.dims.cpu().numpy()       # (l=x, w=y, h=z)
    yaws = pred.bboxes_3d.yaw.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy().astype(int)

    records: list[dict] = []
    for c, dim, yaw, score, label in zip(centers, dims, yaws, scores, labels):
        if score < score_threshold:
            continue
        center_ego = rot_cs @ c.astype(np.float64) + trans_cs
        box_rot_ego = rot_cs @ _yaw_rotation(float(yaw))
        yaw_ego = float(np.arctan2(box_rot_ego[1, 0], box_rot_ego[0, 0]))
        length, width, height = float(dim[0]), float(dim[1]), float(dim[2])
        records.append({
            "center_ego": [float(center_ego[0]), float(center_ego[1]), float(center_ego[2])],
            "yaw": yaw_ego,
            "size": [width, length, height],  # our Box3D order: width, length, height
            "detection_name": NUSC_CLASSES[label] if 0 <= label < len(NUSC_CLASSES) else str(label),
            "detection_score": float(score),
        })
    return records


def _strip_multisweep(model) -> None:
    pipe = model.cfg.test_dataloader.dataset.pipeline
    model.cfg.test_dataloader.dataset.pipeline = [t for t in pipe if t["type"] != "LoadPointsFromMultiSweeps"]


def export_scene(
    dataroot: str | Path | None,
    config: str | Path,
    checkpoint: str | Path,
    scene_index: int,
    max_frames: int | None,
    output_path: str | Path,
    score_threshold: float = 0.1,
    sweeps: int = 10,
    device: str = "cuda:0",
) -> Path:
    from mmdet3d.apis import inference_detector, init_model
    from nuscenes.nuscenes import NuScenes

    loader = NuScenesSceneLoader(dataroot=dataroot)
    scene = loader.scenes[scene_index]
    nusc = NuScenes(version=loader.version, dataroot=str(loader.dataroot), verbose=False)

    model = init_model(str(config), str(checkpoint), device=device)
    if sweeps > 1:
        _strip_multisweep(model)

    samples: dict[str, list[dict]] = {}
    token = scene["first_sample_token"]
    n = 0
    tmp = os.path.join(tempfile.gettempdir(), "centerpoint_sweep.bin")
    while token and (max_frames is None or n < max_frames):
        sample = nusc.get("sample", token)
        pts, ref_cs = _aggregate_sweeps(nusc, sample, nsweeps=max(sweeps, 1))
        pts.tofile(tmp)

        out = inference_detector(model, tmp)
        result = out[0] if isinstance(out, tuple) else out
        if isinstance(result, (list, tuple)):
            result = result[0]
        records = _boxes_to_ego_records(result.pred_instances_3d, ref_cs, score_threshold)
        samples[token] = records
        n += 1
        print(f"frame {n}: {len(records)} boxes ({pts.shape[0]} pts)")
        token = sample["next"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"samples": samples}, handle)
    print(f"wrote {output} ({n} frames, {sum(len(v) for v in samples.values())} boxes)")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CenterPoint 3D detections to prediction JSON.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--config", required=True, help="mmdet3d CenterPoint config .py path.")
    parser.add_argument("--checkpoint", required=True, help="CenterPoint checkpoint .pth path.")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--sweeps", type=int, default=10, help="LiDAR sweeps to accumulate (nuScenes default 10).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/centerpoint_scene0.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_scene(
        dataroot=args.dataroot,
        config=args.config,
        checkpoint=args.checkpoint,
        scene_index=args.scene_index,
        max_frames=args.frames,
        output_path=args.output,
        score_threshold=args.score_threshold,
        sweeps=args.sweeps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
