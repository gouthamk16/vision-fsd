"""planner_bev: one planning step as a paper-style figure.

Left, the scene around the ego: HD map, LiDAR, detected vehicles with forecasts, a sample of the rejected
trajectories, the human's logged future and the chosen plan. Right, the front camera with both paths projected
onto the road, and the learned cost at 1, 2 and 3 s, the way the NMP paper presents its cost volume. Drawn with
matplotlib for real typography, legend and colour bar; everything is vector or rasterised at output resolution,
never upscaled.
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from fsd.bev import BevGrid
from fsd.contact_sheet import read_image
from fsd.data import CameraFrame, SurroundFrame
from fsd.nmp.metrics import ego_footprint
from fsd.nmp.planner import PlanResult
from fsd.nuscenes_map import NuScenesMapRenderer

# Ego-centred and biased forward, like NAVSIM's 64 m frame. 20 px/m is above the figure's ~14 px/m,
# so the map raster is only ever downsampled.
CROP = BevGrid(x_range=(-10.0, 40.0), y_range=(-24.0, 24.0), resolution=0.05)
COST_SECONDS = (1, 2, 3)
SHOWN_CANDIDATES = 40
# The ego frame's z = 0 is the road under the rear axle.
GROUND_Z = 0.4

PLAN, HUMAN, CANDIDATE, VEHICLE, EGO = "#2ca02c", "#ff7f0e", "#9db4cc", "#1f77b4", "#2b2f36"
GROUND, OBSTACLE = "#c9ced5", "#59616d"
# RGB, drawn in order. Kept grey so the planner's output carries all the colour.
MAP_STYLE = {
    "drivable_area": (236, 238, 241),
    "ped_crossing": (222, 226, 231),
    "walkway": (246, 246, 246),
    "lane_divider": (184, 190, 198),
    "road_divider": (138, 147, 158),
}
HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]
LINE_HALO = [patheffects.Stroke(linewidth=5.5, foreground="white"), patheffects.Normal()]
EGO_BODY = ego_footprint(np.zeros(3))


def _uv(xy: np.ndarray) -> np.ndarray:
    """Ego (x forward, y left) -> plot (right, up)."""
    return np.stack([-xy[..., 1], xy[..., 0]], axis=-1)


def _path(trajectory: np.ndarray) -> np.ndarray:
    return _uv(np.vstack([[0.0, 0.0], trajectory[:, :2]]))


def _extent(grid: BevGrid) -> tuple[float, float, float, float]:
    return -grid.y_range[1], -grid.y_range[0], grid.x_range[0], grid.x_range[1]


def _quiet(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c8c8c8")


def _style_axes(ax) -> None:
    ax.set_xlim(-CROP.y_range[1], -CROP.y_range[0])
    ax.set_ylim(*CROP.x_range)
    ax.set_aspect("equal")
    _quiet(ax)


def _dense(trajectory: np.ndarray) -> np.ndarray:
    """Path from the rear axle through `trajectory`, resampled 20x so it can be clipped at the camera."""
    xy = np.vstack([[0.0, 0.0], trajectory[:, :2]])
    t = np.linspace(0, len(xy) - 1, 20 * (len(xy) - 1) + 1)
    return np.column_stack([np.interp(t, np.arange(len(xy)), xy[:, i]) for i in (0, 1)])


def _camera_pixels(xy: np.ndarray, frame: SurroundFrame, camera: CameraFrame) -> np.ndarray:
    """(N, 2) pixels of ego-frame road points (z = 0) in `camera`; points under 1 m in front of it are dropped."""
    local = camera.calibration.invert(camera.ego_pose.invert(frame.ego_pose.apply(np.column_stack([xy, np.zeros(len(xy))]))))
    projected = local[local[:, 2] > 1.0] @ camera.intrinsic.T
    return projected[:, :2] / projected[:, 2:3]


def _draw_camera(ax, frame: SurroundFrame, result: PlanResult, logged: np.ndarray | None) -> None:
    _quiet(ax)
    camera = frame.cameras.get("CAM_FRONT")
    if camera is None:
        ax.text(0.5, 0.5, "no front camera for this frame", ha="center", va="center", transform=ax.transAxes, color="#888888")
        return
    image = read_image(camera)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.plot(*_camera_pixels(_dense(result.plan), frame, camera).T, color=PLAN, lw=3.5, solid_capstyle="round", path_effects=LINE_HALO)
    ax.plot(*_camera_pixels(result.plan[:, :2], frame, camera).T, ls="none", marker="o", ms=5, mfc="white", mec=PLAN, mew=1.4)
    if logged is not None:
        ax.plot(*_camera_pixels(_dense(logged), frame, camera).T, color=HUMAN, lw=2.0, ls=(0, (3, 2)))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title("front camera  ·  plan and human path projected onto the road", fontsize=9, loc="left")


def _draw_scene(ax, frame: SurroundFrame, points: np.ndarray, map_renderer: NuScenesMapRenderer | None) -> None:
    if map_renderer is not None:
        image = np.full((*CROP.shape, 3), 255, np.uint8)
        for mask, color in zip(map_renderer.layer_masks(frame, CROP, tuple(MAP_STYLE)).astype(bool), MAP_STYLE.values()):
            image[mask] = color
        ax.imshow(image, extent=_extent(CROP), zorder=0)
    lo, hi = np.array([CROP.x_range[0], CROP.y_range[0]]), np.array([CROP.x_range[1], CROP.y_range[1]])
    points = points[np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)]
    for keep, color, size in ((points[:, 2] < GROUND_Z, GROUND, 0.3), (points[:, 2] >= GROUND_Z, OBSTACLE, 0.6)):
        ax.scatter(*_uv(points[keep, :2]).T, s=size, c=color, linewidths=0, rasterized=True, zorder=1)


def _draw_agents(ax, result: PlanResult) -> None:
    for box, track in zip(result.detections, result.forecasts):
        ax.add_patch(Polygon(_uv(box.footprint), closed=True, fc=to_rgba(VEHICLE, 0.18), ec=VEHICLE, lw=1.1, zorder=3))
        ax.plot(*_uv(track[:, :2]).T, color=VEHICLE, lw=0.9, alpha=0.75, marker="o", ms=1.8, zorder=3)


def _draw_paths(ax, result: PlanResult, logged: np.ndarray | None) -> None:
    step = max(1, len(result.candidates) // SHOWN_CANDIDATES)
    for trajectory in result.candidates[::step]:
        ax.plot(*_path(trajectory).T, color=CANDIDATE, lw=0.6, alpha=0.7, zorder=2)
    ax.add_patch(Polygon(_uv(EGO_BODY), fc=EGO, ec="white", lw=0.8, zorder=5))
    plan = _path(result.plan)
    ax.plot(*plan.T, color=PLAN, lw=3.0, solid_capstyle="round", path_effects=LINE_HALO, zorder=6)
    if logged is not None:  # thin and on top, so it stays visible where it overlaps the plan
        ax.plot(*_path(logged).T, color=HUMAN, lw=1.6, ls=(0, (3, 2)), zorder=6.5)
    ax.plot(*plan[1:].T, ls="none", marker="o", ms=4.5, mfc="white", mec=PLAN, mew=1.4, zorder=7)
    length = float(np.linalg.norm(np.diff(plan, axis=0), axis=1).sum())
    labels = [(f"{s} s", 2 * s) for s in COST_SECONDS] if length >= 4.0 else [("stopped" if length < 0.5 else "3 s", -1)]
    for text, k in labels:
        ax.annotate(text, plan[k], xytext=(7, -3), textcoords="offset points", fontsize=8, color=PLAN, weight="bold", path_effects=HALO, zorder=8)


def _draw_cost(ax, result: PlanResult, cost_grid: BevGrid, second: int, norm: Normalize) -> None:
    k = 2 * second - 1  # steps are 0.5 s apart
    ax.imshow(result.cost_volume[k], extent=_extent(cost_grid), cmap="magma", norm=norm, zorder=0)
    ax.add_patch(Polygon(_uv(EGO_BODY), fill=False, ec="white", lw=0.9, zorder=1))
    ax.plot(*_path(result.plan[: k + 1]).T, color=PLAN, lw=1.6, zorder=1)
    ax.plot(*_uv(result.plan[k, :2]), marker="o", ms=6, mfc=PLAN, mec="white", mew=1.2, zorder=2)
    _style_axes(ax)
    ax.set_title(f"learned cost, t = {second} s", fontsize=8.5, loc="left")


def _cost_norm(volume: np.ndarray, grid: BevGrid) -> Normalize:
    """Shared colour scale from robust percentiles inside the crop, so the three panels compare."""
    r0, r1 = (int((grid.x_range[1] - x) / grid.resolution) for x in (CROP.x_range[1], CROP.x_range[0]))
    c0, c1 = (int((grid.y_range[1] - y) / grid.resolution) for y in (CROP.y_range[1], CROP.y_range[0]))
    lo, hi = np.percentile(volume[:, r0:r1, c0:c1], (2, 98))
    return Normalize(float(lo), float(max(hi, lo + 1e-6)))


def _annotate(ax, frame: SurroundFrame, result: PlanResult, logged: np.ndarray | None) -> None:
    length = float(np.linalg.norm(np.diff(_path(result.plan), axis=0), axis=1).sum())
    stats = f"plan {length:.1f} m in 3 s"
    if logged is not None and len(logged) == len(result.plan):
        stats += f"  ·  {np.hypot(*(result.plan[-1, :2] - logged[-1, :2])):.1f} m from the human at 3 s"
    ax.set_title(f"{frame.scene_name}  ·  frame {frame.sample_index}  ·  ego {result.v0:.1f} m/s", fontsize=10.5, loc="left")
    ax.set_title(stats, fontsize=9, loc="right", color="#555555")
    handles = [Line2D([], [], color=PLAN, lw=3, marker="o", mfc="white", mec=PLAN, label="NMP plan (dots every 0.5 s)")]
    if logged is not None:
        handles.append(Line2D([], [], color=HUMAN, lw=1.6, ls=(0, (3, 2)), label="human driver (logged)"))
    handles += [
        Line2D([], [], color=CANDIDATE, lw=1, label=f"rejected samples ({SHOWN_CANDIDATES} of {len(result.candidates)})"),
        Patch(fc=to_rgba(VEHICLE, 0.18), ec=VEHICLE, label="detected vehicle + 3 s forecast"),
        Patch(fc=EGO, label="ego"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92, edgecolor="#d0d0d0")
    bar = AnchoredSizeBar(ax.transData, 10, "10 m", "lower right", frameon=False, size_vertical=0.25, fontproperties=FontProperties(size=8))
    ax.add_artist(bar)


def render_planner_bev(
    frame: SurroundFrame,
    points: np.ndarray,
    result: PlanResult,
    cost_grid: BevGrid,
    map_renderer: NuScenesMapRenderer | None = None,
    logged: np.ndarray | None = None,
) -> np.ndarray:
    """BGR image; `logged` is the human's future (T, 3) in this frame's ego frame, when known."""
    fig = Figure(figsize=(16, 9), dpi=100, facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.02), left=0.01, right=0.965, top=0.94, bottom=0.025, wspace=0.04)
    right = outer[1].subgridspec(2, 1, height_ratios=(1.0, 0.68), hspace=0.12)
    costs = right[1].subgridspec(1, 4, width_ratios=(1, 1, 1, 0.06), wspace=0.08)
    ax = fig.add_subplot(outer[0])
    _draw_scene(ax, frame, points, map_renderer)
    _draw_agents(ax, result)
    _draw_paths(ax, result, logged)
    _style_axes(ax)
    _annotate(ax, frame, result, logged)
    _draw_camera(fig.add_subplot(right[0]), frame, result, logged)
    norm = _cost_norm(result.cost_volume, cost_grid)
    for col, second in enumerate(COST_SECONDS):
        panel = fig.add_subplot(costs[col])
        _draw_cost(panel, result, cost_grid, second, norm)
    cax = fig.add_subplot(costs[3])
    panel.apply_aspect()  # equal aspect shrinks the panels; match the colour bar to their real height
    cax.set_position([cax.get_position().x0, panel.get_position().y0, cax.get_position().width, panel.get_position().height])
    bar = fig.colorbar(ScalarMappable(norm, "magma"), cax=cax)
    bar.set_label("learned cost (dark = preferred)", fontsize=9)
    bar.ax.tick_params(labelsize=7)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
