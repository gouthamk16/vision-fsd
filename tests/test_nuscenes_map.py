import json
import math

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.bev import BevGrid
from fsd.nuscenes_map import NuScenesMapRenderer


def _write_map(root, square_center):
    cx, cy = square_center
    corners = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx + 1, cy + 1), (cx - 1, cy + 1)]
    data = {
        "node": [{"token": f"n{i}", "x": x, "y": y} for i, (x, y) in enumerate(corners)],
        "polygon": [{"token": "p", "exterior_node_tokens": ["n0", "n1", "n2", "n3"], "holes": []}],
        "line": [],
        "drivable_area": [{"polygon_tokens": ["p"]}],
        "walkway": [], "ped_crossing": [], "road_divider": [], "lane_divider": [],
    }
    (root / "maps" / "expansion").mkdir(parents=True)
    (root / "maps" / "expansion" / "boston-seaport.json").write_text(json.dumps(data))


def test_features_in_the_view_corners_are_drawn_when_rotated(tmp_path):
    # Ego yaw 45 deg: the view corner (49, 49) in the ego frame is ~69 m straight
    # along global +y, outside the old 60 m culling radius.
    _write_map(tmp_path, (0.0, 49 * math.sqrt(2)))
    grid = BevGrid(resolution=0.5)
    image = NuScenesMapRenderer(tmp_path).render(make_frame(yaw_pose(yaw=math.pi / 4)), grid)
    rows, cols, inside = grid.cells(np.array([[49.0, 49.0]]))
    assert inside[0]
    assert tuple(image[rows[0], cols[0]]) != (248, 248, 248)
