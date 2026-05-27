"""Render nuScenes HD vector maps into an ego-frame BEV canvas."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fsd.data import NuScenesSceneLoader, SurroundFrame
from fsd.lidar_projection import quaternion_to_rotation_matrix


# BGR. LSS paper used RGB (1.00, 0.50, 0.31) for road/lane fill,
# (0.0, 0.0, 1.0) for road_divider, (159/255, 0.0, 1.0) for lane_divider.
ROAD_FILL_BGR = (79, 128, 255)
ROAD_DIVIDER_BGR = (255, 0, 0)
LANE_DIVIDER_BGR = (255, 0, 159)
PED_CROSSING_BGR = (160, 100, 160)


class NuScenesMapRenderer:
    """Render top-down ego-frame views of nuScenes HD maps.

    Loads `maps/expansion/<location>.json` lazily per location. For each
    polygon and line we also cache its global-frame bounding box so the
    per-frame query stays cheap.
    """

    def __init__(
        self,
        dataroot: str | Path | None = None,
        scene_loader: NuScenesSceneLoader | None = None,
    ):
        if scene_loader is None:
            scene_loader = NuScenesSceneLoader(dataroot=dataroot)
        self.scene_loader = scene_loader
        self.expansion_dir = scene_loader.dataroot / "maps" / "expansion"
        if not self.expansion_dir.exists():
            raise FileNotFoundError(
                f"nuScenes map expansion folder not found: {self.expansion_dir}. "
                "Download nuScenes-map-expansion-v1.3 and unzip to maps/."
            )

        logs = scene_loader._load_table("log.json")
        log_by_token = {log["token"]: log for log in logs}
        self.scene_location = {
            scene["token"]: log_by_token[scene["log_token"]]["location"]
            for scene in scene_loader.scenes
        }

        self._cached: dict[str, dict] = {}

    def location_for_scene(self, scene_token: str) -> str:
        return self.scene_location[scene_token]

    def _load(self, location: str) -> dict:
        if location in self._cached:
            return self._cached[location]

        path = self.expansion_dir / f"{location}.json"
        if not path.exists():
            raise FileNotFoundError(f"Map file missing: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        nodes = {n["token"]: (n["x"], n["y"]) for n in data["node"]}

        polygons: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for poly in data["polygon"]:
            tokens = poly.get("exterior_node_tokens") or []
            if len(tokens) < 3:
                continue
            pts = np.array([nodes[t] for t in tokens], dtype=np.float32)
            polygons[poly["token"]] = (pts, pts.min(axis=0), pts.max(axis=0))

        lines: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for line in data["line"]:
            tokens = line.get("node_tokens") or []
            if len(tokens) < 2:
                continue
            pts = np.array([nodes[t] for t in tokens], dtype=np.float32)
            lines[line["token"]] = (pts, pts.min(axis=0), pts.max(axis=0))

        cached = {
            "polygons": polygons,
            "lines": lines,
            "road_segment": data.get("road_segment", []),
            "lane": data.get("lane", []),
            "road_divider": data.get("road_divider", []),
            "lane_divider": data.get("lane_divider", []),
            "ped_crossing": data.get("ped_crossing", []),
            "walkway": data.get("walkway", []),
        }
        self._cached[location] = cached
        return cached

    def render(
        self,
        frame: SurroundFrame,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        output_hw: tuple[int, int] | None = None,
        background: tuple[int, int, int] = (248, 248, 248),
        road_alpha: float = 0.32,
        line_thickness: int = 1,
    ) -> np.ndarray:
        location = self.location_for_scene(frame.scene_token)
        idx = self._load(location)

        ego = frame.ego_pose
        ego_xy = np.asarray(ego["translation"][:2], dtype=np.float32)
        rot_2d = quaternion_to_rotation_matrix(ego["rotation"])[:2, :2].astype(np.float32)

        if output_hw is None:
            h = int(round((x_range[1] - x_range[0]) / 0.25))
            w = int(round((y_range[1] - y_range[0]) / 0.25))
        else:
            h, w = output_hw
        x_max = float(x_range[1])
        y_max = float(y_range[1])
        x_span = float(x_range[1] - x_range[0])
        y_span = float(y_range[1] - y_range[0])

        canvas = np.full((h, w, 3), background, dtype=np.uint8)
        road_mask = np.zeros((h, w), dtype=np.uint8)

        stretch = max(x_span, y_span) / 2 + 10.0
        global_lo = ego_xy - stretch
        global_hi = ego_xy + stretch

        def global_to_pixel(pts: np.ndarray) -> np.ndarray:
            local = (pts - ego_xy) @ rot_2d
            cols = (y_max - local[:, 1]) * (w / y_span)
            rows = (x_max - local[:, 0]) * (h / x_span)
            return np.stack([cols, rows], axis=1).astype(np.int32)

        def bbox_overlaps(pmin, pmax) -> bool:
            return not (
                pmax[0] < global_lo[0]
                or pmin[0] > global_hi[0]
                or pmax[1] < global_lo[1]
                or pmin[1] > global_hi[1]
            )

        # Road + lane polygons -> alpha-blended fill via a mask.
        for layer in ("road_segment", "lane"):
            for item in idx[layer]:
                poly = idx["polygons"].get(item.get("polygon_token"))
                if poly is None:
                    continue
                pts, pmin, pmax = poly
                if not bbox_overlaps(pmin, pmax):
                    continue
                pixels = global_to_pixel(pts).reshape(-1, 1, 2)
                cv2.fillPoly(road_mask, [pixels], 255)

        if road_mask.any():
            road_color = np.array(ROAD_FILL_BGR, dtype=np.float32)[None, None, :]
            mask_f = (road_mask > 0)[..., None].astype(np.float32) * road_alpha
            canvas = (canvas.astype(np.float32) * (1 - mask_f) + road_color * mask_f).astype(np.uint8)

        # Pedestrian crossings — light overlay on top of the road fill.
        ped_mask = np.zeros((h, w), dtype=np.uint8)
        for item in idx["ped_crossing"]:
            poly = idx["polygons"].get(item.get("polygon_token"))
            if poly is None:
                continue
            pts, pmin, pmax = poly
            if not bbox_overlaps(pmin, pmax):
                continue
            pixels = global_to_pixel(pts).reshape(-1, 1, 2)
            cv2.fillPoly(ped_mask, [pixels], 255)
        if ped_mask.any():
            ped_color = np.array(PED_CROSSING_BGR, dtype=np.float32)[None, None, :]
            mask_f = (ped_mask > 0)[..., None].astype(np.float32) * 0.35
            canvas = (canvas.astype(np.float32) * (1 - mask_f) + ped_color * mask_f).astype(np.uint8)

        # Road and lane dividers — crisp lines on top.
        for item in idx["road_divider"]:
            line = idx["lines"].get(item.get("line_token"))
            if line is None:
                continue
            pts, pmin, pmax = line
            if not bbox_overlaps(pmin, pmax):
                continue
            pixels = global_to_pixel(pts).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pixels], False, ROAD_DIVIDER_BGR, line_thickness, cv2.LINE_AA)

        for item in idx["lane_divider"]:
            line = idx["lines"].get(item.get("line_token"))
            if line is None:
                continue
            pts, pmin, pmax = line
            if not bbox_overlaps(pmin, pmax):
                continue
            pixels = global_to_pixel(pts).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pixels], False, LANE_DIVIDER_BGR, line_thickness, cv2.LINE_AA)

        return canvas
