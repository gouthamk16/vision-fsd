"""Render the nuScenes HD map (map expansion v1.3) into ego-frame BEV views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fsd.bev import SHIFT, BevGrid, canvas, fixed_point
from fsd.data import SurroundFrame

# Channel order of `layer_masks`, the planner's map input.
MAP_LAYERS = ("drivable_area", "lane", "ped_crossing", "walkway", "stop_line", "carpark_area", "road_divider", "lane_divider")
_LINE_LAYERS = {"road_divider", "lane_divider"}
_POLYGON_LAYERS = ("walkway", "ped_crossing", "lane", "stop_line", "carpark_area")

# (fill layers, line layers) drawn in order; BGR.
_PALETTES = {
    "light": (
        {"drivable_area": (190, 214, 252), "walkway": (226, 226, 226), "ped_crossing": (206, 178, 206)},
        {"road_divider": (255, 0, 0), "lane_divider": (255, 0, 159)},
        (248, 248, 248),
    ),
    "dark": (
        {"drivable_area": (56, 48, 40), "walkway": (40, 40, 40), "ped_crossing": (80, 60, 80)},
        {"road_divider": (215, 150, 70), "lane_divider": (120, 120, 120)},
        (20, 20, 20),
    ),
}


@dataclass(frozen=True)
class _Layer:
    shapes: list[list[np.ndarray]]  # per item: exterior (+ holes) or a single polyline, global xy
    bounds: np.ndarray              # (N, 4) x_min, y_min, x_max, y_max


class NuScenesMapRenderer:
    """Loads `maps/expansion/<location>.json` lazily and draws its layers per frame."""

    def __init__(self, dataroot: str | Path):
        self.expansion_dir = Path(dataroot) / "maps" / "expansion"
        if not self.expansion_dir.exists():
            raise FileNotFoundError(
                f"nuScenes map expansion folder not found: {self.expansion_dir}. "
                "Download nuScenes-map-expansion-v1.3 and unzip to maps/."
            )
        self._layers: dict[str, dict[str, _Layer]] = {}

    def render(self, frame: SurroundFrame, grid: BevGrid, dark: bool = False) -> np.ndarray:
        image = canvas(grid, _PALETTES["dark" if dark else "light"][2])
        self.draw(image, frame, grid, dark)
        return image

    def draw(self, image: np.ndarray, frame: SurroundFrame, grid: BevGrid, dark: bool = False, fill: bool = True) -> None:
        fills, lines, _ = _PALETTES["dark" if dark else "light"]
        visible = self._visible(frame, grid)
        if fill:
            for name, color in fills.items():
                for contours in visible(name):
                    cv2.fillPoly(image, contours, color, cv2.LINE_AA, SHIFT)
        for name, color in lines.items():
            for (line,) in visible(name):
                cv2.polylines(image, [line], False, color, 1, cv2.LINE_AA, SHIFT)

    def layer_masks(self, frame: SurroundFrame, grid: BevGrid, layers: tuple[str, ...] = MAP_LAYERS) -> np.ndarray:
        """Binary (len(layers), H, W) masks; lines are drawn 2 px wide so they survive coarse grids."""
        visible = self._visible(frame, grid)
        masks = np.zeros((len(layers), *grid.shape), dtype=np.uint8)
        for mask, name in zip(masks, layers):
            for shapes in visible(name):
                if name in _LINE_LAYERS:
                    cv2.polylines(mask, shapes, False, 1, 2, cv2.LINE_8, SHIFT)
                else:
                    cv2.fillPoly(mask, shapes, 1, cv2.LINE_8, SHIFT)
        return masks

    def _visible(self, frame: SurroundFrame, grid: BevGrid):
        """Generator factory: a layer's shapes overlapping the view, in fixed-point grid pixels."""
        layers = self._load(frame.location)
        rot = frame.ego_pose.rotation[:2, :2]
        origin = frame.ego_pose.translation[:2]
        corners = np.array([[x, y] for x in grid.x_range for y in grid.y_range]) @ rot.T + origin
        lo, hi = corners.min(axis=0), corners.max(axis=0)

        def visible(name: str):
            layer = layers[name]
            hit = (layer.bounds[:, 0] <= hi[0]) & (layer.bounds[:, 2] >= lo[0]) & (layer.bounds[:, 1] <= hi[1]) & (layer.bounds[:, 3] >= lo[1])
            for i in np.flatnonzero(hit):
                yield [fixed_point(grid, (pts - origin) @ rot) for pts in layer.shapes[i]]

        return visible

    def _load(self, location: str) -> dict[str, _Layer]:
        if location not in self._layers:
            path = self.expansion_dir / f"{location}.json"
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            nodes = {n["token"]: (n["x"], n["y"]) for n in data["node"]}
            polygons = {p["token"]: p for p in data["polygon"]}
            lines = {line["token"]: line for line in data["line"]}

            def polygon(token: str) -> list[np.ndarray]:
                poly = polygons[token]
                rings = [poly["exterior_node_tokens"], *(hole["node_tokens"] for hole in poly.get("holes", []))]
                return [np.array([nodes[t] for t in ring]) for ring in rings if len(ring) >= 3]

            items = {name: [polygon(r["polygon_token"]) for r in data.get(name, [])] for name in _POLYGON_LAYERS}
            items["drivable_area"] = [polygon(t) for record in data["drivable_area"] for t in record["polygon_tokens"]]
            for name in _LINE_LAYERS:
                items[name] = [[np.array([nodes[t] for t in lines[r["line_token"]]["node_tokens"]])] for r in data.get(name, [])]
            self._layers[location] = {name: _layer([s for s in shapes if s and len(s[0]) >= 2]) for name, shapes in items.items()}
        return self._layers[location]


def _layer(shapes: list[list[np.ndarray]]) -> _Layer:
    bounds = np.array([[*s[0].min(axis=0), *s[0].max(axis=0)] for s in shapes]).reshape(-1, 4)
    return _Layer(shapes, bounds)
