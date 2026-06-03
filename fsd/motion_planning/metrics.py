from __future__ import annotations

from fsd.motion_planning.state import PlanningResult
from fsd.motion_planning.trajectory import trajectory_path_length


def planning_metrics(result: PlanningResult) -> dict[str, float]:
    return {
        "candidate_count": float(len(result.candidates)),
        "valid_candidate_count": float(len(result.valid_candidates)),
        "selected_cost": float(result.selected.cost),
        "selected_path_length_m": float(trajectory_path_length(result.selected.points)),
        "emergency_stop": 1.0 if result.selected.is_emergency_stop else 0.0,
        "fallback_initial_speed": 1.0 if result.world.ego.fallback_initial_speed else 0.0,
    }
