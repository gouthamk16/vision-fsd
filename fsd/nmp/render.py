"""planner_bev: LiDAR + learned cost + sampled and selected trajectories + detections."""

from __future__ import annotations

import cv2
import numpy as np

from fsd.bev import SHIFT, BevGrid, draw_ego, draw_points, fixed_point, lidar_canvas, resample
from fsd.data import SurroundFrame
from fsd.draw import blend, colorize, put_text, with_header
from fsd.nmp.planner import PlanResult
from fsd.object_detection import draw_boxes_on_bev

# Magenta appears nowhere else in the view, so the chosen plan can't be confused with boxes or the ego.
PLAN_COLOR = (255, 0, 255)
CANDIDATE_COLOR = (120, 120, 120)
FORECAST_COLOR = (40, 220, 255)
SHOWN_CANDIDATES = 30


def _polyline(image: np.ndarray, view: BevGrid, xy: np.ndarray, color, thickness: int) -> None:
    pts = fixed_point(view, np.vstack([[0.0, 0.0], xy]))
    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA, SHIFT)


def _cost_heat(image: np.ndarray, result: PlanResult, cost_grid: BevGrid, view: BevGrid) -> np.ndarray:
    """Cheapest cost over the horizon, brightest where the planner would most like to drive."""
    best = result.cost_volume.min(axis=0)
    attraction = 1.0 - (best - best.min()) / max(float(np.ptp(best)), 1e-6)
    covered = resample(np.ones_like(best, np.float32), cost_grid, view, cv2.INTER_NEAREST)
    heat = resample(colorize(attraction, 0.0, 1.0, cv2.COLORMAP_INFERNO), cost_grid, view)
    return blend(image, heat, 0.5 * covered)


def _draw_plan(image: np.ndarray, view: BevGrid, plan: np.ndarray) -> None:
    """Drawn over the ego: when stopped, the whole 3 s plan lies inside the ego footprint."""
    _polyline(image, view, plan[:, :2], (0, 0, 0), 7)
    _polyline(image, view, plan[:, :2], PLAN_COLOR, 4)
    draw_points(image, view, plan[:, :2], np.tile(np.array(PLAN_COLOR, np.uint8), (len(plan), 1)), radius=4)
    end = (fixed_point(view, plan[-1:, :2])[0, 0] >> SHIFT) + (22, 5)  # clear of the ego heading arrow
    put_text(image, "plan 3s", tuple(int(v) for v in end), 0.5, PLAN_COLOR)


def render_planner_bev(
    frame: SurroundFrame,
    points: np.ndarray,
    result: PlanResult,
    cost_grid: BevGrid,
    view: BevGrid,
    background: np.ndarray | None = None,
) -> np.ndarray:
    image = _cost_heat(lidar_canvas(points, view, background), result, cost_grid, view)
    step = max(1, len(result.candidates) // SHOWN_CANDIDATES)
    for trajectory in result.candidates[::step]:
        _polyline(image, view, trajectory[:, :2], CANDIDATE_COLOR, 1)
    draw_boxes_on_bev(image, view, result.detections)
    if len(result.forecasts):
        dots = result.forecasts[:, 1:, :2].reshape(-1, 2)
        draw_points(image, view, dots, np.tile(np.array(FORECAST_COLOR, np.uint8), (len(dots), 1)), radius=2)
    draw_ego(image, view)
    _draw_plan(image, view, result.plan)
    length = float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], result.plan[:, :2]]), axis=0), axis=1).sum())
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | neural motion planner | v0 {result.v0:.1f} m/s",
        f"MAGENTA = chosen plan, {length:.1f} m in 3 s (dot every 0.5 s) | gray = {SHOWN_CANDIDATES} of {len(result.candidates)} "
        f"candidates it rejected | green = {len(result.detections)} detected cars, yellow = where they go",
    )
