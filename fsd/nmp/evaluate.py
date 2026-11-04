"""Validation metrics for the planner and the `eval` report against baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fsd.nmp.config import NmpConfig
from fsd.nmp.dataset import make_loader
from fsd.nmp.metrics import REPORT_SECONDS, PlanMetrics, constant_velocity_plan
from fsd.nmp.model import NeuralMotionPlanner
from fsd.nmp.planner import decode_detections, select_plan
from fsd.nmp.samples import CLASS_NAMES
from fsd.nmp.trajectories import sample_trajectories

MATCH_DISTANCE_M = 2.0
# nuScenes detection eval: cars count within 50 m. AP ranks every detection, so decode far below the view's 0.3.
MAX_RANGE_M = 50.0
AP_THRESHOLD = 0.05
# Khurana et al., ECCV 2022, Tab. 4: simplified NMP (no map cost) on nuScenes-val.
REFERENCE = {"l2": (0.53, 1.25, 2.67), "collision": (0.04, 0.12, 0.87)}


def _plans(model: NeuralMotionPlanner, loader, cfg: NmpConfig):
    """Yield (batch on CPU as numpy, model output, selected plans) per batch."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["input"].to(device)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            candidates = sample_trajectories(batch["v0"].to(device), batch["k0"].to(device), cfg, cfg.n_trajectories, generator=generator)
            plans = [candidates[b, select_plan(out.cost[b].float(), candidates[b], cfg.grid)[0]].cpu().numpy() for b in range(len(x))]
            yield {k: v.numpy() for k, v in batch.items() if k != "input"}, out, plans


def evaluate(model: NeuralMotionPlanner, loader, cfg: NmpConfig) -> dict[str, float]:
    metrics = PlanMetrics()
    for batch, _, plans in _plans(model, loader, cfg):
        for b, plan in enumerate(plans):
            metrics.add(plan, batch["ego_future"][b], batch["boxes"][b], batch["box_valid"][b], batch["drivable"][b], cfg)
    return metrics.summary()


def _match(boxes, forecasts, gt_boxes, gt_valid, gt_class, gt_points, cfg: NmpConfig):
    """Greedy distance matching at step 0: (scores, true-positive flags, forecast errors (n, 3), n_gt).

    Like the nuScenes detection eval, ground truth the sensor never saw (no LiDAR points) is not a target,
    and neither is ground truth outside the model's grid.
    """
    detect = [CLASS_NAMES.index(c) for c in cfg.det_classes]
    near = np.hypot(gt_boxes[:, 0, 0], gt_boxes[:, 0, 1]) <= MAX_RANGE_M
    on_grid = cfg.grid.cells(gt_boxes[:, 0, :2])[2]
    gt_rows = np.flatnonzero(gt_valid[:, 0] & np.isin(gt_class, detect) & near & on_grid & (gt_points > 0))
    free = set(gt_rows.tolist())
    scores, tps, errors = [], [], []
    for i in np.argsort([-b.score for b in boxes]):
        if np.hypot(*forecasts[i, 0, :2]) > MAX_RANGE_M:
            continue
        dists = {r: float(np.hypot(*(gt_boxes[r, 0, :2] - forecasts[i, 0, :2]))) for r in free}
        row = min(dists, key=dists.get) if dists else None
        hit = row is not None and dists[row] < MATCH_DISTANCE_M
        scores.append(boxes[i].score)
        tps.append(hit)
        if hit:
            free.discard(row)
            steps = [2 * s for s in REPORT_SECONDS]
            errors.append([np.hypot(*(gt_boxes[row, k, :2] - forecasts[i, k, :2])) if gt_valid[row, k] else np.nan for k in steps])
    return scores, tps, errors, len(gt_rows)


def average_precision(scores: list[float], tps: list[bool], n_gt: int) -> float:
    """All-point interpolated AP over a ranked detection list."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    tp = np.cumsum(np.array(tps, float)[order])
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * envelope))


def report(weights: Path, cache_dir: Path, split: str, cfg: NmpConfig) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralMotionPlanner(cfg).to(device)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    nmp, cv, expert = PlanMetrics(), PlanMetrics(), PlanMetrics()
    scores, tps, errors, n_gt, travel = [], [], [], 0, []
    for batch, out, plans in _plans(model, make_loader(cache_dir, split, cfg, 4, 4, None, False), cfg):
        boxes, forecasts = decode_detections(out, cfg, AP_THRESHOLD)
        for b, plan in enumerate(plans):
            args = (batch["ego_future"][b], batch["boxes"][b], batch["box_valid"][b], batch["drivable"][b], cfg)
            nmp.add(plan, *args)
            cv.add(constant_velocity_plan(float(batch["v0"][b]), cfg), *args)
            expert.add(batch["ego_future"][b], *args)
            travel.append((np.hypot(*plan[-1, :2]), np.hypot(*batch["ego_future"][b][-1, :2])))
            s, t, e, n = _match(boxes[b], forecasts[b], batch["boxes"][b], batch["box_valid"][b], batch["box_class"][b], batch["box_points"][b], cfg)
            scores, tps, errors, n_gt = scores + s, tps + t, errors + e, n_gt + n
    rows = {"expert (logged)": expert.summary(), "CV baseline": cv.summary(), "NMP (ours)": nmp.summary()}
    planned, logged = np.mean(travel, axis=0)
    table = _table(rows, average_precision(scores, tps, n_gt), np.array(errors).reshape(-1, 3))
    return f"{table}\nmean distance from start at 3 s: NMP {planned:.2f} m | expert {logged:.2f} m"


def _table(rows: dict[str, dict], ap: float, errors: np.ndarray) -> str:
    """Rows of PlanMetrics summaries; the expert row is the floor the metrics allow, not a competitor."""
    cols = [f"{m} {s}s" for m in ("L2 (m)", "coll (%)", "off-road (%)") for s in REPORT_SECONDS]
    lines = [f"| method | {' | '.join(cols)} |", "|---" * (len(cols) + 1) + "|"]
    for name, row in rows.items():
        values = [row[f"{m}_{s}s"] for m in ("l2", "collision", "offroad") for s in REPORT_SECONDS]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    ref = [*REFERENCE["l2"], *REFERENCE["collision"], *("-",) * 3]
    lines.append("| NMP ref (Khurana et al. ECCV'22, nuScenes-val, no map) | " + " | ".join(str(v) for v in ref) + " |")
    forecast = ", ".join(f"{s}s {np.nanmean(errors[:, i]):.2f} m" for i, s in enumerate(REPORT_SECONDS)) if len(errors) else "n/a"
    lines += ["", f"samples: {int(rows['NMP (ours)']['samples'])} | vehicle AP @ {MATCH_DISTANCE_M:.0f} m within {MAX_RANGE_M:.0f} m: {ap:.3f} | forecast L2 of matched: {forecast}"]
    return "\n".join(lines)
