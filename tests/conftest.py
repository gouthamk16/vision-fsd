import math
from pathlib import Path

import numpy as np
import pytest

from fsd.data import Pose, SurroundFrame


def yaw_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Pose:
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))


def make_frame(pose: Pose, timestamp_us: int = 0, scene_token: str = "scene") -> SurroundFrame:
    return SurroundFrame(scene_token, "scene-x", "boston-seaport", "sample", 0, timestamp_us, pose, {})


@pytest.fixture
def tmp_json(tmp_path: Path):
    def write(text: str) -> Path:
        path = tmp_path / "table.json"
        path.write_text(text, encoding="utf-8")
        return path

    return write
