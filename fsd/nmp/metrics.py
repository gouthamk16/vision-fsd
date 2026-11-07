"""Open-loop planning metrics: L2 to the driven trajectory, box collision and off-road rates."""

from __future__ import annotations

import numpy as np

from fsd.ego_motion import EGO_FOOTPRINT
from fsd.nmp.config import NmpConfig
from fsd.nmp.dataset import footprint

REPORT_SECONDS = (1, 2, 3)


def boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex (4, 2) rectangles."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def ego_footprint(pose: np.ndarray) -> np.ndarray:
    """(4, 2) ego body corners at a rear-axle pose (x, y, yaw), in Box3D corner order."""
    (x0, x1), (y0, y1) = EGO_FOOTPRINT
    local = np.array([[x1, y1], [x1, y0], [x0, y0], [x0, y1]])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return local @ np.array([[c, -s], [s, c]]).T + pose[:2]


def constant_velocity_plan(v0: float, cfg: NmpConfig) -> np.ndarray:
    plan = np.zeros((cfg.horizon, 3), np.float32)
    plan[:, 0] = v0 * cfg.times
    return plan


class PlanMetrics:
    """Accumulates per-sample results; rates are cumulative up to each reported second."""

    def __init__(self):
        self.l2, self.collided, self.offroad = [], [], []

    def add(self, plan, ego_future, boxes, box_valid, drivable, cfg: NmpConfig) -> None:
        self.l2.append(np.linalg.norm(plan[:, :2] - ego_future[:, :2], axis=1))
        hits, off = np.zeros(cfg.horizon, bool), np.zeros(cfg.horizon, bool)
        rows, cols, inside = cfg.grid.cells(np.stack([ego_footprint(p).mean(axis=0) for p in plan]))
        for k, pose in enumerate(plan):
            body = ego_footprint(pose)
            hits[k] = any(boxes_overlap(body, footprint(boxes[i, k + 1])) for i in np.flatnonzero(box_valid[:, k + 1]))
            off[k] = inside[k] and not drivable[rows[k], cols[k]]  # past the grid the map can't judge
        self.collided.append(np.logical_or.accumulate(hits))
        self.offroad.append(np.logical_or.accumulate(off))

    def summary(self) -> dict[str, float]:
        l2, collided, offroad = np.array(self.l2), np.array(self.collided), np.array(self.offroad)
        result = {"samples": float(len(l2))}
        for sec in REPORT_SECONDS:
            k = 2 * sec - 1  # steps are 0.5 s apart
            result[f"l2_{sec}s"] = float(l2[:, k].mean())
            result[f"collision_{sec}s"] = 100.0 * float(collided[:, k].mean())
            result[f"offroad_{sec}s"] = 100.0 * float(offroad[:, k].mean())
        return result
