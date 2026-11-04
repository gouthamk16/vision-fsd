"""planner_bev: LiDAR + learned cost + sampled and selected trajectories + detections."""

from __future__ import annotations

import cv2
import numpy as np

from fsd.bev import SHIFT, BevGrid, draw_ego, draw_points, fixed_point, lidar_canvas, resample
from fsd.data import SurroundFrame
from fsd.draw import blend, colorize, with_header
from fsd.nmp.planner import PlanResult
from fsd.object_detection import draw_boxes_on_bev

PLAN_COLOR = (80, 255, 80)
CANDIDATE_COLOR = (150, 150, 150)
FORECAST_COLOR = (40, 220, 255)
SHOWN_CANDIDATES = 100


def _polyline(image: np.ndarray, view: BevGrid, xy: np.ndarray, color, thickness: int) -> None:
    pts = fixed_point(view, np.vstack([[0.0, 0.0], xy]))
    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA, SHIFT)


def render_planner_bev(
    frame: SurroundFrame,
    points: np.ndarray,
    result: PlanResult,
    cost_grid: BevGrid,
    view: BevGrid,
    background: np.ndarray | None = None,
) -> np.ndarray:
    image = lidar_canvas(points, view, background)
    # Cheapest cost over the horizon, brightest where the planner would most like to drive.
    best = result.cost_volume.min(axis=0)
    attraction = 1.0 - (best - best.min()) / max(float(np.ptp(best)), 1e-6)
    covered = resample(np.ones_like(best, np.float32), cost_grid, view, cv2.INTER_NEAREST)
    heat = resample(colorize(attraction, 0.0, 1.0, cv2.COLORMAP_INFERNO), cost_grid, view)
    image = blend(image, heat, 0.5 * covered)

    step = max(1, len(result.candidates) // SHOWN_CANDIDATES)
    for trajectory in result.candidates[::step]:
        _polyline(image, view, trajectory[:, :2], CANDIDATE_COLOR, 1)
    draw_boxes_on_bev(image, view, result.detections)
    if len(result.forecasts):
        dots = result.forecasts[:, 1:, :2].reshape(-1, 2)
        draw_points(image, view, dots, np.tile(np.array(FORECAST_COLOR, np.uint8), (len(dots), 1)), radius=2)
    _polyline(image, view, result.plan[:, :2], PLAN_COLOR, 3)
    draw_points(image, view, result.plan[:, :2], np.tile(np.array(PLAN_COLOR, np.uint8), (len(result.plan), 1)), radius=4)
    draw_ego(image, view)
    plan_cost = float(result.costs.min())
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | neural motion planner | v0 {result.v0:.1f} m/s",
        f"bright = low learned cost | gray = {SHOWN_CANDIDATES} of {len(result.candidates)} samples | green = plan (dots every 0.5 s), "
        f"cost {plan_cost:.0f} | {len(result.detections)} detections",
    )
